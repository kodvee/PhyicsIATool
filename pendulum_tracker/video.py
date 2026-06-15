"""Video access helpers.

A 4K/60fps source is far too large to hold in memory, so every frame is read
on demand from disk via a single cached ``cv2.VideoCapture`` handle and frames
are only ever decoded one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


class VideoReader:
    """Thin wrapper around ``cv2.VideoCapture`` with random frame access.

    Seeking with ``CAP_PROP_POS_FRAMES`` is exact enough for our purposes and
    avoids decoding the whole file. The handle is reused across calls.
    """

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {path}")
        self._last_seek = -1

    def info(self) -> VideoInfo:
        return VideoInfo(
            path=self.path,
            fps=float(self.cap.get(cv2.CAP_PROP_FPS)) or 60.0,
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read_frame(self, index: int) -> np.ndarray | None:
        """Return the BGR frame at ``index`` or ``None`` if it cannot be read."""
        index = max(0, int(index))
        # Avoid a redundant seek when reading sequentially.
        if index != self._last_seek + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        self._last_seek = index if ok else -1
        return frame if ok else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):  # best-effort cleanup
        try:
            self.release()
        except Exception:
            pass


def downscale_for_display(frame: np.ndarray, max_width: int = 960):
    """Downscale a (possibly 4K) BGR frame for browser display.

    Returns ``(rgb_small, scale)`` where multiplying a coordinate in the
    displayed image by ``scale`` maps it back to full-resolution pixels.
    """
    h, w = frame.shape[:2]
    scale = 1.0
    if w > max_width:
        scale = w / float(max_width)
        new_size = (max_width, int(round(h / scale)))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb, scale
