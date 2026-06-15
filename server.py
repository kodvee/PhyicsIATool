"""Pendulum Decay Video Analyzer — Flask wizard backend.

A local single-user utility. State lives in one in-process object (no DB, no
sessions). The UI is a staged wizard: each stage is its own page/route and the
user cannot jump to a stage that has not been unlocked yet.

Run:  python server.py    (or ./run.sh)
"""

from __future__ import annotations

import os
import tempfile
import threading

import cv2
import numpy as np
from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from pendulum_tracker.calibration import Calibration, scale_from_points
from pendulum_tracker.export import ExportMeta, export_folder, run_dir, write_run_csv
from pendulum_tracker.process import TrackConfig, diagnostic_sample, process_run
from pendulum_tracker.proxy import RunClipExporter, ffmpeg_available
from pendulum_tracker.tracking import HSVRange, crop_roi, detect_bob
from pendulum_tracker.video import VideoReader

app = Flask(__name__)

DISPLAY_WIDTH = 960   # max width of frames shown in the browser
DISPLAY_HEIGHT = 620  # max height — keeps a tall/portrait clip on screen

# Ordered wizard stages: (slug, title). Index in this list == stage number - 1.
STAGES = [
    ("video", "Video & Metadata"),
    ("runs", "Runs"),
    ("roi", "Colour"),
    ("calibration", "Scale & Origin"),
    ("diagnostics", "Tracking Health"),
    ("filter", "Noise Filter"),
    ("process", "Process & Export"),
    ("results", "Results"),
]


class AppState:
    """Everything the wizard accumulates for the current video file."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.reader: VideoReader | None = None
        self.lock = threading.Lock()  # cv2.VideoCapture is not thread-safe
        self.info = None
        self.video_path = ""
        self.gap_height = ""
        self.pendulum_L = 0.0
        self.mode = "video"                  # "video" | "runs_folder"
        self.runs = []                       # [{name,start,end,baseline[,clip_path,clip_frames,clip_fps]}]
        self.hsv = HSVRange().__dict__.copy()
        self.min_area = 30
        self.scale_S = None
        self.origin = None                   # (x,y) full-frame px
        self.filter_method = "butterworth"
        self.cutoff_hz = 3.0
        self.ma_window = 7
        self.results = {}                    # run_name -> DataFrame
        self.export_dir = os.path.join(os.getcwd(), "exports")
        # Display-only viewport crop (x,y,w,h) in full-frame px. Shrinks the
        # shown image (e.g. for a tall portrait clip) so the pendulum and the
        # controls fit on screen together. Tracking/calibration still use true
        # full-frame coordinates, so this never affects the physics.
        self.view_crop = None
        self.clip_exporter = RunClipExporter()  # full-res per-run clip exporter
        self.run_dirs = {}                   # run name -> output folder
        self.max_unlocked = 1                # highest stage number reachable

    @property
    def view(self) -> tuple[int, int, int, int]:
        """Current viewport crop in full-frame px (full frame if unset)."""
        if self.view_crop:
            return self.view_crop
        if self.info:
            return (0, 0, self.info.width, self.info.height)
        return (0, 0, 0, 0)

    def hsv_obj(self) -> HSVRange:
        return HSVRange(**self.hsv)

    def unlock(self, stage_number: int):
        self.max_unlocked = max(self.max_unlocked, stage_number)

    def read_frame(self, idx: int):
        with self.lock:
            if self.reader is None:
                return None
            return self.reader.read_frame(idx)


STATE = AppState()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fit_jpeg(bgr, max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT, quality=80):
    """Uniformly scale a frame to fit within (max_width, max_height) and encode.

    Uniform scaling keeps the px→full-frame mapping a single factor, which the
    frontend recovers as sourceWidth / displayedWidth.
    """
    h, w = bgr.shape[:2]
    s = min(max_width / w, max_height / h, 1.0)
    if s < 1.0:
        bgr = cv2.resize(bgr, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def _require_video():
    if STATE.reader is None or STATE.info is None:
        abort(400, "No video loaded.")


# --------------------------------------------------------------------------- #
# Page routes (the wizard)
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return redirect(url_for("stage", n=1))


@app.route("/stage/<int:n>")
def stage(n: int):
    if n < 1 or n > len(STAGES):
        abort(404)
    if n > STATE.max_unlocked:
        # Cannot skip ahead — bounce to the furthest unlocked stage.
        return redirect(url_for("stage", n=STATE.max_unlocked))
    slug, title = STAGES[n - 1]
    return render_template(
        f"stage_{slug}.html",
        n=n,
        slug=slug,
        title=title,
        stages=STAGES,
        max_unlocked=STATE.max_unlocked,
        state=STATE,
        info=STATE.info,
        view=STATE.view,
        display_width=DISPLAY_WIDTH,
    )


# --------------------------------------------------------------------------- #
# Frame / image endpoints
# --------------------------------------------------------------------------- #
@app.route("/api/frame/<int:idx>")
def api_frame(idx):
    """Frame for display. Cropped to the viewport unless ?raw=1 (crop tool)."""
    _require_video()
    frame = STATE.read_frame(idx)
    if frame is None:
        abort(404, "Frame not readable.")
    if not request.args.get("raw"):
        frame = crop_roi(frame, STATE.view)
        if frame.size == 0:
            abort(400, "Empty viewport crop.")
    data = _fit_jpeg(frame)
    if data is None:
        abort(500, "Encode failed.")
    from io import BytesIO

    return send_file(BytesIO(data), mimetype="image/jpeg")


@app.route("/api/mask")
def api_mask():
    """Red-mask preview of the cropped detection region at a given frame.

    The detection region is the spatial crop (STATE.view) — there is no longer
    a separate ROI rectangle; tracking runs directly on the cropped video.
    """
    _require_video()
    idx = int(request.args.get("frame", 0))
    frame = STATE.read_frame(idx)
    if frame is None:
        abort(404)
    crop = crop_roi(frame, STATE.view)
    if crop.size == 0:
        abort(400, "Empty crop.")
    mask = STATE.hsv_obj().mask(crop)
    bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    data = _fit_jpeg(bgr, max_width=480, max_height=480)
    from io import BytesIO

    return send_file(BytesIO(data), mimetype="image/jpeg")


@app.route("/api/detect")
def api_detect():
    """Bob detection status at a frame, in full-frame pixel coords."""
    _require_video()
    idx = int(request.args.get("frame", 0))
    frame = STATE.read_frame(idx)
    if frame is None:
        abort(404)
    crop = crop_roi(frame, STATE.view)
    res = detect_bob(crop, STATE.hsv_obj(), STATE.min_area, tuple(STATE.view[:2]))
    return jsonify(found=res.found, point=res.point)


# --------------------------------------------------------------------------- #
# Stage save endpoints
# --------------------------------------------------------------------------- #
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "pendulum_uploads")


def _open_active(path: str):
    """Open ``path`` as the active source used for display/calibration."""
    with STATE.lock:
        if STATE.reader:
            STATE.reader.release()
        STATE.reader = VideoReader(path)
        STATE.info = STATE.reader.info()
    STATE.video_path = path


def _info_dict():
    i = STATE.info
    return dict(width=i.width, height=i.height, fps=i.fps,
               frame_count=i.frame_count, duration=i.duration_s)


def _finalize_load(path: str, gap_height: str, pendulum_L):
    """Open a single video for the normal (crop + run-select) workflow."""
    if not path or not os.path.exists(path):
        return jsonify(ok=False, error="Video file not found."), 400
    try:
        _open_active(path)
        STATE.mode = "video"
        STATE.runs = []
        STATE.gap_height = (gap_height or "").strip()
        STATE.pendulum_L = float(pendulum_L or 0.0)
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=str(e)), 400
    if STATE.info.frame_count <= 0:
        return jsonify(ok=False, error="Could not read any frames from this file."), 400
    if not STATE.gap_height:
        return jsonify(ok=False, error="Target gap height is required."), 400
    STATE.view_crop = None  # reset viewport to the full frame
    STATE.unlock(2)
    return jsonify(ok=True, mode="video", info=_info_dict(), view=STATE.view)


@app.route("/api/load_video", methods=["POST"])
def api_load_video():
    """Load a video already on disk by path (used by tests / power users)."""
    data = request.get_json(force=True)
    return _finalize_load(
        (data.get("path") or "").strip(),
        data.get("gap_height"), data.get("pendulum_L"),
    )


@app.route("/api/upload_video", methods=["POST"])
def api_upload_video():
    """Receive a browser file upload (streamed to disk) and load it."""
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(ok=False, error="No file uploaded."), 400
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    name = secure_filename(f.filename) or "upload.mp4"
    dest = os.path.join(UPLOAD_DIR, name)
    f.save(dest)  # werkzeug streams large files to disk rather than memory
    return _finalize_load(
        dest, request.form.get("gap_height"), request.form.get("pendulum_L")
    )


@app.route("/api/upload_runs_folder", methods=["POST"])
def api_upload_runs_folder():
    """Load a folder of already-saved run clips (e.g. a prior height_<gap>mm
    export). Each video file becomes one run; the first clip is the active
    source for colour/scale/origin calibration (all clips share the same crop)."""
    files = request.files.getlist("files")
    vids = [f for f in files if f.filename and
            os.path.splitext(f.filename)[1].lower() in (".mp4", ".mov", ".avi", ".mkv")]
    if not vids:
        return jsonify(ok=False, error="No video files found in that folder."), 400
    dest_root = os.path.join(UPLOAD_DIR, "runs_" + str(int(os.times().elapsed)))
    os.makedirs(dest_root, exist_ok=True)
    # Sort by the (relative) path the browser reports so run order is stable.
    vids.sort(key=lambda f: f.filename)

    runs, saved = [], []
    for i, f in enumerate(vids, 1):
        base = os.path.basename(f.filename)
        dest = os.path.join(dest_root, secure_filename(f"{i:02d}_{base}"))
        f.save(dest)
        try:
            r = VideoReader(dest)
            inf = r.info()
            r.release()
        except Exception:  # noqa: BLE001
            continue
        if inf.frame_count <= 0:
            continue
        # Name from a "runN" parent folder if present, else the file stem.
        parent = os.path.basename(os.path.dirname(f.filename))
        stem = parent if parent.lower().startswith("run") else os.path.splitext(base)[0]
        runs.append(dict(name=stem or f"run{i}", start=0, end=inf.frame_count - 1,
                         baseline=False, clip_path=dest,
                         clip_frames=inf.frame_count, clip_fps=inf.fps))
        saved.append(dest)

    if not runs:
        return jsonify(ok=False, error="Could not read any of the run videos."), 400

    # De-duplicate run names (folders may repeat).
    seen = {}
    for r in runs:
        n = r["name"]
        if n in seen:
            seen[n] += 1
            r["name"] = f"{n}_{seen[n]}"
        else:
            seen[n] = 1

    try:
        _open_active(runs[0]["clip_path"])     # first clip drives the previews
        STATE.mode = "runs_folder"
        STATE.runs = runs
        STATE.view_crop = None
        STATE.gap_height = (request.form.get("gap_height") or "").strip()
        STATE.pendulum_L = float(request.form.get("pendulum_L") or 0.0)
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=str(e)), 400
    if not STATE.gap_height:
        return jsonify(ok=False, error="Target gap height is required."), 400
    STATE.unlock(2)
    return jsonify(ok=True, mode="runs_folder", info=_info_dict(), view=STATE.view,
                   runs=[{k: r[k] for k in ("name", "start", "end")} for r in STATE.runs])


@app.route("/api/viewcrop", methods=["POST"])
def api_viewcrop():
    """Set the display viewport crop (full-frame px). Display-only; the physics
    keeps using true full-frame coordinates."""
    _require_video()
    d = request.get_json(force=True)
    if d.get("reset"):
        STATE.view_crop = None
        return jsonify(ok=True, view=STATE.view)
    try:
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
    except Exception:  # noqa: BLE001
        return jsonify(ok=False, error="Bad crop rectangle."), 400
    W, H = STATE.info.width, STATE.info.height
    x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    if w < 16 or h < 16:
        return jsonify(ok=False, error="Crop is too small."), 400
    STATE.view_crop = (x, y, w, h)
    return jsonify(ok=True, view=STATE.view)


@app.route("/api/runs", methods=["GET", "POST", "DELETE"])
def api_runs():
    if request.method == "GET":
        return jsonify(runs=STATE.runs)
    if request.method == "DELETE":
        name = request.get_json(force=True).get("name")
        STATE.runs = [r for r in STATE.runs if r["name"] != name]
        return jsonify(ok=True, runs=STATE.runs)
    # POST add run — names are auto-assigned (lowest free runN).
    if STATE.mode == "runs_folder":
        return jsonify(ok=False, error="Runs come from the loaded folder."), 400
    d = request.get_json(force=True)
    try:
        start, end = int(d["start"]), int(d["end"])
        baseline = bool(d.get("baseline"))
    except Exception:  # noqa: BLE001
        return jsonify(ok=False, error="Invalid run fields."), 400
    if end <= start:
        return jsonify(ok=False, error="End frame must exceed start frame."), 400
    existing = {r["name"] for r in STATE.runs}
    i = 1
    while f"run{i}" in existing:
        i += 1
    name = f"run{i}"
    if baseline:  # only one baseline allowed
        for r in STATE.runs:
            r["baseline"] = False
    STATE.runs.append(dict(name=name, start=start, end=end, baseline=baseline))
    return jsonify(ok=True, runs=STATE.runs)


@app.route("/api/advance/<int:to_stage>", methods=["POST"])
def api_advance(to_stage):
    """Unlock the next stage after validating the prerequisite for it."""
    # Validate what is needed to ENTER `to_stage`.
    checks = {
        3: (bool(STATE.runs), "Add at least one run first."),
        5: (STATE.scale_S is not None and STATE.origin is not None,
            "Complete scale and origin calibration first."),
        8: (bool(STATE.results), "Process the runs first."),
    }
    ok, msg = checks.get(to_stage, (True, ""))
    if not ok:
        return jsonify(ok=False, error=msg), 400
    STATE.unlock(to_stage)
    return jsonify(ok=True, next=url_for("stage", n=to_stage))


@app.route("/api/hsv", methods=["POST"])
def api_hsv():
    d = request.get_json(force=True)
    for k in ("h_center", "h_tol", "s_min", "s_max", "v_min", "v_max"):
        if k in d:
            STATE.hsv[k] = int(d[k])
    if "min_area" in d:
        STATE.min_area = int(d["min_area"])
    return jsonify(ok=True, hsv=STATE.hsv, min_area=STATE.min_area)


@app.route("/api/scale", methods=["POST"])
def api_scale():
    d = request.get_json(force=True)
    try:
        p1 = (float(d["x1"]), float(d["y1"]))
        p2 = (float(d["x2"]), float(d["y2"]))
        dist = float(d["distance_mm"])
        STATE.scale_S = scale_from_points(p1, p2, dist)
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, scale=STATE.scale_S)


@app.route("/api/origin", methods=["POST"])
def api_origin():
    d = request.get_json(force=True)
    try:
        STATE.origin = (float(d["x"]), float(d["y"]))
    except Exception:  # noqa: BLE001
        return jsonify(ok=False, error="Bad origin."), 400
    return jsonify(ok=True, origin=STATE.origin)


@app.route("/api/diagnostic", methods=["POST"])
def api_diagnostic():
    _require_video()
    d = request.get_json(force=True)
    run = d.get("run")
    cfg = TrackConfig(STATE.view, STATE.hsv_obj(), STATE.min_area)
    if run and run != "__all__":
        r = next((r for r in STATE.runs if r["name"] == run), None)
        if not r:
            return jsonify(ok=False, error="Unknown run."), 400
        if r.get("clip_path"):  # folder mode: each run is its own clip
            rd = VideoReader(r["clip_path"])
            try:
                hits, total, _ = diagnostic_sample(rd, 0, rd.info().frame_count - 1, cfg, 100)
            finally:
                rd.release()
        else:
            with STATE.lock:
                hits, total, _ = diagnostic_sample(STATE.reader, r["start"], r["end"], cfg, 100)
    else:
        with STATE.lock:
            hits, total, _ = diagnostic_sample(STATE.reader, 0, STATE.info.frame_count - 1, cfg, 100)
    pct = (hits / total * 100) if total else 0
    return jsonify(ok=True, hits=hits, total=total, pct=pct)


@app.route("/api/filter", methods=["POST"])
def api_filter():
    d = request.get_json(force=True)
    STATE.filter_method = d.get("method", "butterworth")
    STATE.cutoff_hz = float(d.get("cutoff_hz", 3.0))
    STATE.ma_window = int(d.get("ma_window", 7))
    return jsonify(ok=True)


@app.route("/api/process", methods=["POST"])
def api_process():
    _require_video()
    d = request.get_json(force=True)
    STATE.export_dir = (d.get("export_dir") or STATE.export_dir).strip()
    if STATE.scale_S is None or STATE.origin is None:
        return jsonify(ok=False, error="Calibration incomplete."), 400
    if not STATE.runs:
        return jsonify(ok=False, error="No runs defined."), 400

    calib = Calibration(STATE.scale_S, STATE.origin)
    cfg = TrackConfig(STATE.view, STATE.hsv_obj(), STATE.min_area)
    folder = export_folder(STATE.export_dir, STATE.gap_height)
    meta = ExportMeta(STATE.gap_height, STATE.pendulum_L, STATE.scale_S)

    STATE.results = {}
    STATE.run_dirs = {}
    summary = []
    for r in STATE.runs:
        if r.get("clip_path"):  # folder mode: process the run's own clip
            rd_reader = VideoReader(r["clip_path"])
            try:
                inf = rd_reader.info()
                df = process_run(
                    rd_reader, 0, inf.frame_count - 1, inf.fps, cfg, calib,
                    STATE.cutoff_hz, STATE.filter_method, STATE.ma_window,
                )
            finally:
                rd_reader.release()
        else:
            with STATE.lock:
                df = process_run(
                    STATE.reader, r["start"], r["end"], STATE.info.fps, cfg, calib,
                    STATE.cutoff_hz, STATE.filter_method, STATE.ma_window,
                )
        rd = run_dir(folder, r["name"])           # own folder per run
        path = write_run_csv(rd, r["name"], df, meta)
        STATE.results[r["name"]] = df
        STATE.run_dirs[r["name"]] = rd
        detected = int(df["X_raw (mm)"].notna().sum())
        summary.append(dict(name=r["name"], frames=len(df), detected=detected, path=path))
    STATE.unlock(8)
    return jsonify(ok=True, folder=folder, summary=summary)


@app.route("/api/export_videos", methods=["POST"])
def api_export_videos():
    """Save each run's full-res, full-fps cropped clip into its own folder.

    Triggered right after the Runs stage. In folder mode the run videos already
    exist, so this is a no-op.
    """
    _require_video()
    if not STATE.runs:
        return jsonify(ok=False, error="No runs defined."), 400
    d = request.get_json(silent=True) or {}
    STATE.export_dir = (d.get("export_dir") or STATE.export_dir).strip()
    folder = export_folder(STATE.export_dir, STATE.gap_height)
    STATE.run_dirs = {r["name"]: run_dir(folder, r["name"]) for r in STATE.runs}

    if STATE.mode == "runs_folder":
        # Clips already exist — copy them into their run folders (no re-encode),
        # keeping each clip's original container extension.
        out_paths = {
            r["name"]: os.path.join(
                STATE.run_dirs[r["name"]],
                r["name"] + (os.path.splitext(r["clip_path"])[1] or ".mp4"),
            )
            for r in STATE.runs
        }
        STATE.clip_exporter.copy_all(STATE.runs, out_paths)
        return jsonify(ok=True)

    if not ffmpeg_available():
        return jsonify(ok=False, error="ffmpeg is not installed on PATH."), 400
    out_paths = {
        r["name"]: os.path.join(STATE.run_dirs[r["name"]], f"{r['name']}.mp4")
        for r in STATE.runs
    }
    STATE.clip_exporter.export_all(
        STATE.video_path, STATE.view, STATE.info.fps, STATE.runs, out_paths
    )
    return jsonify(ok=True)


@app.route("/api/export_status")
def api_export_status():
    return jsonify(STATE.clip_exporter.snapshot())


@app.route("/api/results/<run>")
def api_results(run):
    df = STATE.results.get(run)
    if df is None:
        abort(404)
    out = df.where(df.notna(), None)  # JSON-safe NaN -> null
    return jsonify(
        t=out["Timestamp (s)"].tolist(),
        x_raw=out["X_raw (mm)"].tolist(),
        x_filt=out["X_filtered (mm)"].tolist(),
        y_filt=out["Y_filtered (mm)"].tolist(),
    )


if __name__ == "__main__":
    import webbrowser

    port = int(os.environ.get("PORT", 5000))
    url = f"http://127.0.0.1:{port}"
    print(f"Pendulum Decay Analyzer running at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
