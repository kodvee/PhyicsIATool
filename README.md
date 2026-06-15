# Pendulum Decay Video Analyzer

A lightweight **local** web utility for extracting clean position-vs-time data
from a single long **4K / 60 fps** video that contains several sequential
pendulum-decay runs at one magnet gap height.

Built with a small **Flask** backend + plain **HTML / JS** frontend (no heavy
web framework). The UI is a **staged wizard**: each step is its own full page
and you cannot skip ahead to a stage that has not been unlocked. OpenCV does the
vision, SciPy the filtering, Plotly (via CDN) the graphs, and **ffmpeg** saves
the full-res per-run cropped clips.

## Dependencies

```bash
sudo apt install ffmpeg     # for saving the per-run full-res cropped clips
```

(The analysis and CSV export work without ffmpeg; only "Accept & save full-res
run videos" needs it.)

## Quick start

```bash
./run.sh                 # first run builds the venv, installs deps, opens the browser
# or manually:
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python server.py
```

It serves at http://127.0.0.1:5000 (set `PORT=...` to change) and opens your
browser automatically.

> **4K files are large.** Upload in stage 1 with the file picker (a progress
> bar shows transfer); the upload is streamed to disk and frames are then
> decoded on demand. Since the app runs locally, the upload is just a local
> copy. (A `POST /api/load_video` path-based endpoint also exists for scripting.)

## The 8 stages

1. **Video & Metadata** — either **upload one long clip** (optionally crop it
   spatially to the region of interest) **or load a folder of already-saved run
   videos** to re-analyse directly; enter *Target Gap Height (mm)* and *Measured
   Pendulum Length L (mm)*.
2. **Runs** — play/scrub the timeline in-place; add each run's start/end frame;
   tick exactly one **Baseline** run. On *Continue*, each run's full-resolution,
   full-fps cropped clip is saved to its folder (re-encoded from the source in
   video mode, copied as-is in folder mode).
3. **Colour** — tracking runs **directly on the cropped video** (no extra ROI
   rectangle); tune the red hue with HSV sliders against a live mask + detection
   readout.
4. **Scale & Origin** — click two ruler points + type the real distance → px→mm
   factor `S`; click the resting bob position to set the origin (Y-axis =
   true vertical through it, up = +).
5. **Tracking Health** — samples exactly 100 evenly spaced frames and reports
   e.g. *"Bob detected in 98/100 frames"*.
6. **Noise Filter** — Butterworth low-pass (1–5 Hz cutoff slider) or moving
   average, to strip high-frequency vibration while keeping the slow swing.
7. **Process & Export** — writes `exports/height_<gap>mm/<run>/<run>.csv`, one
   per-run folder, each CSV prefixed with a metadata header (beside the clip).
8. **Results** — interactive Plotly charts (X raw vs filtered, Y filtered), plus
   the status of the run-video save kicked off after stage 2.

Each **selection window** (run scrubber, colour preview, calibration canvases)
has its own in-place **play / pause / frame-step / slider**, so you can play the
video right in the window and pause exactly where you want to make a selection.

### Output layout

```
exports/height_15mm/
  run1/  run1.csv  run1.mp4   ← full-res, full-fps cropped clip of the run
  run2/  run2.csv  run2.mp4
```

## How the bob position is found

The bob carries a red magnet disk; from the side it reads as a short 2D
cylinder. Within the cropped video we take the largest red region and report the
bob position as the **center of its bounding box** (`x = bbox center`,
`y = bbox center`). Using the centre rather than an edge averages the opposing
boundaries, so colour-threshold jitter on a single edge largely cancels — which
matters for the small-amplitude vertical (Y) motion. `X_raw`/`Y_raw` are that
point in mm relative to the origin (Y positive-up). Colour matching runs only on
the frames within each run's start–end range.

## CSV format

```
# Target Gap Height (mm): 15
# Pendulum Length L (mm): 350.0
# Pixel-to-mm Scale Factor (mm/px): 0.50000000
# Run: run1
Frame,Timestamp (s),X_raw (mm),Y_raw (mm),X_filtered (mm),Y_filtered (mm)
...
```

## Project layout

```
server.py                    Flask wizard backend (routes + JSON API + stage gating)
templates/                   one HTML page per wizard stage (+ base.html stepper)
static/                      style.css, common.js (FrameCanvas draw/point helper)
pendulum_tracker/            framework-agnostic core logic:
  video.py                   on-demand frame reader + display downscaling
  tracking.py                HSV red mask + single-bob bottom-line detection
  calibration.py             px→mm scale & origin-centered coordinate transform
  filtering.py               Butterworth / moving-average low-pass
  process.py                 100-frame diagnostic + full-run orchestration
  export.py                  per-run folder + CSV writer with metadata header
  proxy.py                   ffmpeg full-res per-run clip exporter
test_pipeline.py             core logic smoke test on a synthetic clip
test_server.py               full wizard end-to-end test via Flask test client
```

## Tests

```bash
. .venv/bin/activate
python test_pipeline.py      # core math on a synthetic decay clip
python test_server.py        # drives every wizard endpoint end-to-end
```
