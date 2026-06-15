"""Export each run's full-resolution, full-fps cropped clip via the ffmpeg CLI.

These clips keep the native resolution of the cropped region and the source
frame rate, trimmed to a run's frame range, encoded near-lossless. They are
saved next to each run's CSV; the data itself always comes from the
full-resolution source frames, so this never affects the physics.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _even(v: int) -> int:
    """libx264 (yuv420p) needs even dimensions."""
    v = int(round(v))
    return v - (v % 2)


class RunClipExporter:
    """Exports each run's full-resolution, full-fps cropped clip via ffmpeg.

    Clips are NOT downscaled: they keep the native resolution of the cropped
    region and the source frame rate, trimmed to the run's frame range, encoded
    near-lossless (crf 18).
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.status = {"state": "idle", "progress": 0.0, "message": "", "done": []}

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.status)

    def _set(self, **kw):
        with self.lock:
            self.status.update(kw)

    def export_all(self, src, crop, fps, runs, out_paths) -> None:
        """Start a background export. ``runs`` is [{name,start,end},...] and
        ``out_paths`` maps run name -> destination .mp4 path."""
        if self.snapshot()["state"] == "exporting":
            return
        self._set(state="exporting", progress=0.0, message="Starting…", done=[])
        t = threading.Thread(
            target=self._run, args=(src, crop, fps, runs, out_paths), daemon=True
        )
        t.start()

    def copy_all(self, runs, out_paths) -> None:
        """Copy already-encoded run clips into their output folders (folder mode).

        No re-encode — the clips are already the full-res cropped runs.
        ``runs`` carry ``clip_path``; ``out_paths`` maps run name -> destination.
        """
        if self.snapshot()["state"] == "exporting":
            return
        self._set(state="exporting", progress=0.0, message="Copying run videos…", done=[])
        threading.Thread(target=self._copy, args=(runs, out_paths), daemon=True).start()

    def _copy(self, runs, out_paths):
        done, n = [], max(1, len(runs))
        try:
            for i, r in enumerate(runs):
                src, dst = r["clip_path"], out_paths[r["name"]]
                if not (os.path.exists(dst) and os.path.samefile(src, dst)):
                    shutil.copy2(src, dst)
                done.append(dst)
                self._set(done=list(done), progress=(i + 1) / n,
                          message=f"Copied {r['name']} ({i + 1}/{n})")
            self._set(state="ready", progress=1.0, message="Run videos saved.", done=done)
        except Exception as e:  # noqa: BLE001
            self._set(state="error", message=str(e))

    def _run(self, src, crop, fps, runs, out_paths):
        x, y, w, h = crop
        cw, ch = _even(w), _even(h)
        n = max(1, len(runs))
        done = []
        try:
            for i, r in enumerate(runs):
                s, e = int(r["start"]), int(r["end"])
                run_frames = max(1, e - s + 1)
                out = out_paths[r["name"]]
                vf = f"crop={cw}:{ch}:{int(x)}:{int(y)}"
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{s / fps:.6f}", "-t", f"{run_frames / fps:.6f}", "-i", src,
                    "-vf", vf, "-an",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-progress", "pipe:1", "-nostats", "-loglevel", "error",
                    out,
                ]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True, bufsize=1,
                )
                for line in proc.stdout:
                    line = line.strip()
                    if line.startswith("frame="):
                        try:
                            f = int(line.split("=", 1)[1])
                            frac = min(1.0, f / run_frames)
                            self._set(progress=(i + frac) / n,
                                      message=f"Exporting {r['name']} ({i + 1}/{n})…")
                        except ValueError:
                            pass
                    elif line == "progress=end":
                        break
                ret = proc.wait()
                if ret != 0:
                    err = (proc.stderr.read() or "").strip()[:400]
                    self._set(state="error", message=f"ffmpeg failed on {r['name']}: {err or ret}")
                    return
                done.append(out)
                self._set(done=list(done), progress=(i + 1) / n)
            self._set(state="ready", progress=1.0, message="All run videos saved.", done=done)
        except FileNotFoundError:
            self._set(state="error", message="ffmpeg not found on PATH.")
        except Exception as e:  # noqa: BLE001
            self._set(state="error", message=str(e))
