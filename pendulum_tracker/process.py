"""Run-level orchestration: track a frame range and build the result table.

Tracking every frame of a 4K/60fps clip is CPU-bound (HSV conversion, masking,
morphology and contour finding per frame). To use all available cores we split
the frame range into contiguous chunks and hand each chunk to a worker process.
A ``cv2.VideoCapture`` handle is neither thread-safe nor picklable, so every
worker opens its **own** ``VideoReader`` on the same file, seeks once to the
start of its chunk, then reads sequentially. Calibration and filtering run once,
serially, after the per-chunk pixel tracks are stitched back together.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .calibration import Calibration
from .filtering import butterworth_lowpass, moving_average
from .tracking import HSVRange, crop_roi, detect_bob
from .video import VideoReader

# Don't bother spinning up workers for tiny ranges, and keep each chunk large
# enough that the one-off seek to its start frame is amortised over many reads.
_MIN_CHUNK_FRAMES = 150


@dataclass
class TrackConfig:
    roi: tuple[int, int, int, int]
    hsv: HSVRange
    min_area: int


def _track_chunk(args) -> tuple[int, np.ndarray, np.ndarray]:
    """Worker: track frames [start, end] of ``path`` and return raw pixel tracks.

    Runs in a separate process. Returns ``(start, x_px, y_px)`` where the arrays
    cover ``end - start + 1`` frames; missing detections are ``NaN``.
    """
    path, start, end, cfg = args
    n = end - start + 1
    x_px = np.full(n, np.nan)
    y_px = np.full(n, np.nan)
    reader = VideoReader(path)
    try:
        for i in range(n):
            frame = reader.read_frame(start + i)
            if frame is None:
                continue
            roi = crop_roi(frame, cfg.roi)
            res = detect_bob(roi, cfg.hsv, cfg.min_area, cfg.roi[:2])
            if res.found:
                x_px[i], y_px[i] = res.x_raw, res.y_raw
    finally:
        reader.release()
    return start, x_px, y_px


def _chunk_bounds(start: int, end: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split the inclusive range [start, end] into ``n_chunks`` contiguous spans."""
    edges = np.linspace(start, end + 1, n_chunks + 1).astype(int)
    spans = []
    for k in range(n_chunks):
        lo, hi = int(edges[k]), int(edges[k + 1]) - 1
        if hi >= lo:
            spans.append((lo, hi))
    return spans


def _track_range(
    reader: VideoReader,
    start: int,
    end: int,
    cfg: TrackConfig,
    workers: int,
    progress: Callable[[float], None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track [start, end] across ``workers`` processes; return (x_px, y_px).

    Falls back to a single in-process pass when parallelism wouldn't help (a
    short range, ``workers <= 1``, or no usable file path on ``reader``).
    """
    n = end - start + 1
    x_px = np.full(n, np.nan)
    y_px = np.full(n, np.nan)

    path = getattr(reader, "path", None)
    if workers <= 1 or n < 2 * _MIN_CHUNK_FRAMES or not path:
        for i in range(n):
            frame = reader.read_frame(start + i)
            if frame is not None:
                roi = crop_roi(frame, cfg.roi)
                res = detect_bob(roi, cfg.hsv, cfg.min_area, cfg.roi[:2])
                if res.found:
                    x_px[i], y_px[i] = res.x_raw, res.y_raw
            if progress is not None and (i % 20 == 0 or i == n - 1):
                progress((i + 1) / n)
        return x_px, y_px

    # More chunks than workers gives finer progress and better load balancing
    # (a worker that finishes early picks up the next chunk).
    n_chunks = min(workers * 4, max(1, n // _MIN_CHUNK_FRAMES))
    spans = _chunk_bounds(start, end, n_chunks)
    tasks = [(path, lo, hi, cfg) for lo, hi in spans]

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_track_chunk, t) for t in tasks]
        for fut in as_completed(futures):
            chunk_start, xs, ys = fut.result()
            off = chunk_start - start
            x_px[off:off + xs.size] = xs
            y_px[off:off + ys.size] = ys
            done += 1
            if progress is not None:
                progress(done / len(tasks))
    return x_px, y_px


def diagnostic_sample(
    reader: VideoReader,
    start: int,
    end: int,
    cfg: TrackConfig,
    n_samples: int = 100,
):
    """Track ``n_samples`` evenly spaced frames; return (hits, total, frames).

    ``frames`` lists the sampled frame indices where both markers were found,
    useful for spot-checking calibration before committing to a full pass.
    """
    end = max(start, end)
    total = min(n_samples, end - start + 1)
    if total <= 0:
        return 0, 0, []
    indices = np.unique(np.linspace(start, end, total).astype(int))
    hits = 0
    good = []
    for idx in indices:
        frame = reader.read_frame(int(idx))
        if frame is None:
            continue
        roi = crop_roi(frame, cfg.roi)
        res = detect_bob(roi, cfg.hsv, cfg.min_area, cfg.roi[:2])
        if res.found:
            hits += 1
            good.append(int(idx))
    return hits, len(indices), good


def process_run(
    reader: VideoReader,
    start: int,
    end: int,
    fps: float,
    cfg: TrackConfig,
    calib: Calibration,
    cutoff_hz: float,
    filter_method: str = "butterworth",
    ma_window: int = 7,
    progress: Callable[[float], None] | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    """Track every frame in [start, end], calibrate, filter, return a DataFrame.

    ``workers`` sets how many processes share the tracking pass; ``None`` uses
    every available CPU core, ``1`` forces the serial path.
    """
    end = max(start, end)
    frames = list(range(start, end + 1))
    n = len(frames)

    if workers is None:
        workers = os.cpu_count() or 1

    x_px, y_px = _track_range(reader, start, end, cfg, workers, progress)

    # Calibrate raw pixel positions into origin-centered millimetres.
    x_raw_mm, y_raw_mm = calib.to_mm(x_px, y_px)

    if filter_method == "none":
        # No filtering — "filtered" columns are a verbatim copy of raw.
        x_filt, y_filt = x_raw_mm, y_raw_mm
    elif filter_method == "moving_average":
        x_filt = moving_average(x_raw_mm, ma_window)
        y_filt = moving_average(y_raw_mm, ma_window)
    else:
        x_filt = butterworth_lowpass(x_raw_mm, cutoff_hz, fps)
        y_filt = butterworth_lowpass(y_raw_mm, cutoff_hz, fps)

    timestamps = (np.array(frames) - start) / fps

    return pd.DataFrame(
        {
            "Frame": frames,
            "Timestamp (s)": np.round(timestamps, 6),
            "X_raw (mm)": np.round(x_raw_mm, 5),
            "Y_raw (mm)": np.round(y_raw_mm, 5),
            "X_filtered (mm)": np.round(x_filt, 5),
            "Y_filtered (mm)": np.round(y_filt, 5),
        }
    )
