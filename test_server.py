"""Drive the whole Flask wizard end-to-end with the synthetic clip."""

import os
import tempfile

from test_pipeline import make_video
import server


def main():
    tmp = tempfile.mkdtemp(prefix="pendsrv_")
    vid = os.path.join(tmp, "synthetic.mp4")
    make_video(vid)

    c = server.app.test_client()

    # Stage gating: stage 3 must be locked before anything is loaded.
    r = c.get("/stage/3")
    assert r.status_code == 302 and "/stage/1" in r.headers["Location"], "gating failed"

    # 1. upload the video (multipart) like the browser file picker does
    with open(vid, "rb") as fh:
        up = c.post("/api/upload_video", data={
            "file": (fh, "synthetic.mp4"),
            "gap_height": "15", "pendulum_L": "350",
        }, content_type="multipart/form-data")
    j = up.get_json(); assert j["ok"], j
    fc = j["info"]["frame_count"]
    # path-based loading still works (programmatic / tests)
    assert c.post("/api/load_video",
                  json={"path": vid, "gap_height": "15", "pendulum_L": 350}).get_json()["ok"]
    assert j["view"] == [0, 0, j["info"]["width"], j["info"]["height"]]

    # 1b. dimension crop: a sub-rectangle of the frame (display viewport only)
    vc = c.post("/api/viewcrop", json={"x": 100, "y": 80, "w": 400, "h": 300}).get_json()
    assert vc["ok"] and vc["view"] == [100, 80, 400, 300], vc
    # both cropped and raw frames still serve (raw bypasses the viewport crop)
    assert c.get("/api/frame/0").status_code == 200
    assert c.get("/api/frame/0").mimetype == "image/jpeg"
    assert c.get("/api/frame/0?raw=1").status_code == 200

    # crop must not break tracking, which uses true full-frame coords
    vc = c.post("/api/viewcrop", json={"reset": True}).get_json()
    assert vc["ok"]

    # 2. add a run — name is auto-assigned (no name field)
    r = c.post("/api/runs", json={"start": 0, "end": fc - 1, "baseline": True}).get_json()
    assert r["ok"], r
    assert r["runs"][-1]["name"] == "run1", r["runs"]
    assert c.post("/api/advance/3", json={}).get_json()["ok"]

    # 2b. finishing the Runs stage saves the full-res cropped run videos
    import time
    assert c.post("/api/export_videos", json={"export_dir": tmp}).get_json()["ok"]
    deadline = time.time() + 60
    while time.time() < deadline:
        es = c.get("/api/export_status").get_json()
        if es["state"] in ("ready", "error"):
            break
        time.sleep(0.4)
    assert es["state"] == "ready", f"clip export did not finish: {es}"
    clip = es["done"][0]
    assert os.path.exists(clip) and os.path.getsize(clip) > 0, clip
    assert os.path.basename(os.path.dirname(clip)) == "run1", clip
    print("run clip:", clip, os.path.getsize(clip), "bytes")

    # 3. colour only (no ROI rectangle — tracking runs on the cropped video)
    assert c.post("/api/hsv", json={"h_center": 0, "h_tol": 10, "s_min": 120, "s_max": 255,
                                    "v_min": 120, "v_max": 255, "min_area": 20}).get_json()["ok"]
    assert c.get("/api/mask?frame=0").status_code == 200
    det = c.get("/api/detect?frame=0").get_json()
    assert det["found"], det

    # 4. calibration
    assert c.post("/api/scale", json={"x1": 0, "y1": 0, "x2": 100, "y2": 0, "distance_mm": 50}).get_json()["ok"]
    assert c.post("/api/origin", json={"x": 320, "y": 240}).get_json()["ok"]
    assert c.post("/api/advance/5", json={}).get_json()["ok"]

    # 5. diagnostic
    diag = c.post("/api/diagnostic", json={"run": "__all__"}).get_json()
    assert diag["ok"] and diag["hits"] == diag["total"] == 100, diag

    # 6. filter + 7. process — CSV lands in the same per-run folder as the clip
    assert c.post("/api/filter", json={"method": "butterworth", "cutoff_hz": 3.0}).get_json()["ok"]
    assert c.post("/api/advance/7", json={}).get_json()["ok"]
    proc = c.post("/api/process", json={"export_dir": tmp}).get_json()
    assert proc["ok"], proc
    csv_path = proc["summary"][0]["path"]
    assert os.path.exists(csv_path)
    assert os.path.dirname(csv_path) == os.path.dirname(clip), "csv not beside its clip"

    # 8. results JSON for plotting
    res = c.get("/api/results/run1").get_json()
    assert len(res["t"]) == fc and len(res["x_filt"]) == fc
    assert res["x_filt"][0] is not None

    for n in range(1, 9):
        assert c.get(f"/stage/{n}").status_code == 200, f"stage {n} failed to render"

    # --- folder mode: load the exported clip back as a "folder of runs" ---
    with open(clip, "rb") as fh:
        up = c.post("/api/upload_runs_folder", data={
            "files": (fh, "height_15mm/run1/run1.mp4"),
            "gap_height": "15", "pendulum_L": "350",
        }, content_type="multipart/form-data")
    fj = up.get_json()
    assert fj["ok"] and fj["mode"] == "runs_folder", fj
    assert len(fj["runs"]) == 1, fj
    # scale/origin persist; process the folder run directly
    proc2 = c.post("/api/process", json={"export_dir": tmp}).get_json()
    assert proc2["ok"] and proc2["summary"][0]["frames"] > 0, proc2
    # folder mode can ALSO save its run videos — copied (no re-encode) into folders
    assert c.post("/api/export_videos", json={"export_dir": tmp}).get_json()["ok"]
    deadline = time.time() + 30
    while time.time() < deadline:
        es2 = c.get("/api/export_status").get_json()
        if es2["state"] in ("ready", "error"):
            break
        time.sleep(0.3)
    assert es2["state"] == "ready", es2
    assert es2["done"] and os.path.exists(es2["done"][0]), es2
    print("folder-mode run frames:", proc2["summary"][0]["frames"], "saved:", es2["done"][0])

    print("CSV ->", csv_path)
    print("ALL SERVER CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
