"""End-to-end smoke test on a synthetic single-bob decay video.

Generates a small clip with one red bob (a short cylinder/disk) swinging as a
damped sinusoid (plus a high-frequency wobble), then runs tracking, calibration,
filtering and CSV export and checks the numbers are sane.
"""

import os
import tempfile

import cv2
import numpy as np

from pendulum_tracker.calibration import Calibration, scale_from_points
from pendulum_tracker.export import ExportMeta, export_folder, write_run_csv
from pendulum_tracker.process import TrackConfig, diagnostic_sample, process_run
from pendulum_tracker.tracking import HSVRange, crop_roi, detect_bob
from pendulum_tracker.video import VideoReader

BOB_W, BOB_H = 40, 16  # red disk-from-side footprint (px)


def make_video(path, n=240, fps=60, w=640, h=480):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    cx0, cy0 = w // 2, h // 2
    for i in range(n):
        t = i / fps
        swing = 120 * np.exp(-0.4 * t) * np.cos(2 * np.pi * 0.8 * t)   # ~0.8 Hz decay
        wobble = 4 * np.cos(2 * np.pi * 12 * t)                        # 12 Hz wobble
        cx = int(cx0 + swing + wobble)
        cy = cy0
        frame = np.full((h, w, 3), 30, np.uint8)  # dark background
        # single red bob: a short 2D cylinder (BGR red = (0,0,255))
        cv2.rectangle(frame, (cx - BOB_W // 2, cy - BOB_H // 2),
                      (cx + BOB_W // 2, cy + BOB_H // 2), (0, 0, 255), -1)
        vw.write(frame)
    vw.release()


def main():
    tmp = tempfile.mkdtemp(prefix="pendtest_")
    vid = os.path.join(tmp, "synthetic.mp4")
    make_video(vid)

    reader = VideoReader(vid)
    info = reader.info()
    print(f"video: {info.width}x{info.height} {info.fps:.1f}fps {info.frame_count} frames")
    assert info.frame_count >= 230

    hsv = HSVRange(h_center=0, h_tol=10, s_min=120, v_min=120)
    roi = (0, 0, info.width, info.height)
    cfg = TrackConfig(roi, hsv, min_area=20)

    # single-frame detection — point is the bounding-box center of the bob
    f = reader.read_frame(0)
    res = detect_bob(crop_roi(f, roi), hsv, 20, roi[:2])
    assert res.found, "bob not detected on frame 0"
    cx0, cy0 = info.width // 2, info.height // 2
    print(f"frame0 center={res.point}")
    # t=0: swing is at max (cos 0 = 1) plus the +4px wobble => cx0 + 124
    assert abs(res.x_raw - (cx0 + 124)) < 8, "x not at expected bob center"
    assert abs(res.y_raw - cy0) < 4, "y not at bob center"

    # diagnostics
    hits, total, _ = diagnostic_sample(reader, 0, info.frame_count - 1, cfg, 100)
    print(f"diagnostic: {hits}/{total} frames")
    assert hits == total, "expected 100% detection on clean synthetic video"

    # calibration: 100 px == 50 mm  -> S = 0.5
    S = scale_from_points((0, 0), (100, 0), 50.0)
    assert abs(S - 0.5) < 1e-9
    calib = Calibration(S, (info.width / 2, info.height / 2))

    # full process with butterworth @ 3 Hz (should kill 12 Hz wobble)
    df = process_run(reader, 0, info.frame_count - 1, info.fps, cfg, calib,
                     cutoff_hz=3.0, filter_method="butterworth")
    assert len(df) == info.frame_count
    assert not df["X_raw (mm)"].isna().any()

    # filtered std should be < raw std (wobble removed); decay => later amplitude smaller
    raw_hf = np.std(np.diff(df["X_raw (mm)"]))
    filt_hf = np.std(np.diff(df["X_filtered (mm)"]))
    print(f"hf jitter raw={raw_hf:.4f} filtered={filt_hf:.4f}")
    assert filt_hf < raw_hf, "filter did not reduce high-frequency content"

    early = df["X_filtered (mm)"][:60].abs().max()
    late = df["X_filtered (mm)"][-60:].abs().max()
    print(f"amplitude early={early:.2f}mm late={late:.2f}mm")
    assert late < early, "decay envelope not captured"

    # export
    folder = export_folder(tmp, "15")
    meta = ExportMeta("15", 350.0, S)
    out = write_run_csv(folder, "run1", df, meta)
    assert os.path.basename(folder) == "height_15mm"
    with open(out) as fh:
        head = [next(fh) for _ in range(5)]
    assert head[0].startswith("# Target Gap Height (mm): 15")
    assert "Frame,Timestamp (s),X_raw (mm)" in head[4]
    print(f"export OK -> {out}")
    print("header:\n" + "".join(head))

    reader.release()
    print("\nALL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
