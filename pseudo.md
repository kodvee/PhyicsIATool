# Pendulum Bob Tracking Algorithm — Pseudocode

## Overview

The tool processes a video of a pendulum whose bob is marked with a red disk.
For every frame in the selected range it detects the bob's pixel position using
colour segmentation, then converts the pixel track into a physical displacement
time-series (in millimetres) via a spatial calibration, and finally removes
high-frequency noise with a low-pass filter.

---

## Phase 1 — Calibration (run once before tracking)

```
INPUT:  two user-clicked points P1, P2 on a known physical ruler in the frame
        known physical distance D (mm) between P1 and P2
        user-clicked origin point O (resting position of the bob)

PROCEDURE Calibrate():

    1.  COMPUTE pixel distance between P1 and P2:
            pixel_dist = sqrt( (P2.x − P1.x)² + (P2.y − P1.y)² )

    2.  COMPUTE scale factor:
            S = D / pixel_dist          [units: mm per pixel]

    3.  STORE origin O = (O.x, O.y)    [full-frame pixel coordinates]

    RETURN S, O

END PROCEDURE
```

---

## Phase 2 — Per-Frame Bob Detection

```
INPUT:  one video frame (BGR image), region of interest ROI = (x, y, w, h),
        HSV colour thresholds (hue_center, hue_tolerance, s_min, s_max,
        v_min, v_max), minimum blob area A_min

PROCEDURE DetectBob(frame, ROI, HSV_params, A_min):

    ── Step 1: Restrict search area ──────────────────────────────────────────

    1.  CROP frame to ROI boundaries (clamped to frame edges)  →  roi_image
        (Reduces computation and avoids false positives outside the swing area)

    ── Step 2: Colour-space conversion ───────────────────────────────────────

    2.  CONVERT roi_image from BGR to HSV colour space
        (HSV separates hue from brightness, making colour detection more
         robust to changes in lighting intensity)

    ── Step 3: Build red colour mask ─────────────────────────────────────────

    3.  DETERMINE hue window [hue_center − hue_tolerance,
                              hue_center + hue_tolerance]

        IF window wraps below 0°:
            SPLIT into [0°, upper_edge] ∪ [180° + lower_edge, 180°]
        ELSE IF window wraps above 180°:
            SPLIT into [lower_edge, 180°] ∪ [0°, upper_edge − 180°]
        ELSE:
            single range [lower_edge, upper_edge]

        (Red straddles the 0°/180° boundary in OpenCV's hue scale, so
         wrap-around handling is required)

    4.  FOR each hue sub-range [h_lo, h_hi]:
            MARK pixel as 1 if:
                h_lo ≤ pixel.H ≤ h_hi
              AND s_min ≤ pixel.S ≤ s_max
              AND v_min ≤ pixel.V ≤ v_max
            ELSE mark as 0

        COMBINE all sub-range masks with bitwise OR  →  raw_mask

    ── Step 4: Noise removal ─────────────────────────────────────────────────

    5.  APPLY morphological opening to raw_mask using a 3×3 elliptical kernel:
            erosion followed immediately by dilation
        →  clean_mask
        (Removes isolated single-pixel speckles caused by lighting noise or
         colour fringing, while preserving the solid red blob of the bob)

    ── Step 5: Contour detection ─────────────────────────────────────────────

    6.  FIND all external contours (connected white regions) in clean_mask

    7.  DISCARD any contour whose pixel area < A_min
        (Eliminates residual small reflections or background artefacts)

    8.  IF no contours remain:
            RETURN detection_failed  (position = NaN)

    ── Step 6: Select best candidate ─────────────────────────────────────────

    9.  SELECT contour with the largest pixel area  →  best_blob
        (The bob is always the largest red object inside the ROI)

    ── Step 7: Compute bob position ──────────────────────────────────────────

    10. COMPUTE axis-aligned bounding box of best_blob:
            (bx, by, bw, bh)   [in ROI-local pixel coordinates]

    11. COMPUTE bounding-box centre (in full-frame pixel coordinates):
            px = bx + bw / 2  +  ROI.x          ← horizontal midpoint
            py = by + bh / 2  +  ROI.y          ← vertical midpoint

        (Using the centre of the bounding box rather than a single edge
         averages both sides of the disk, so threshold jitter on either edge
         largely cancels — especially important for the small vertical
         displacements that would otherwise appear as a noisy staircase)

    RETURN (px, py)

END PROCEDURE
```

---

## Phase 3 — Full Run: Track All Frames

```
INPUT:  video file, frameStart, frameEnd, ROI, HSV_params, A_min,
        calibration (S, O), frame rate fps,
        low-pass cutoff frequency f_c, filter method

PROCEDURE TrackRun():

    1.  INITIALISE arrays x_px[n], y_px[n]  ← all NaN   (n = frameEnd − frameStart + 1)

    2.  FOR i = 0 TO n − 1:

            frame = READ frame (frameStart + i) from video

            IF frame could not be read:
                CONTINUE  (leave NaN)

            (px, py) = DetectBob(frame, ROI, HSV_params, A_min)

            IF detection succeeded:
                x_px[i] = px
                y_px[i] = py

    END FOR

    ── Calibration ───────────────────────────────────────────────────────────

    3.  CONVERT pixel arrays to millimetres using calibration (S, O):

            X_mm[i] =  (x_px[i] − O.x) × S      [positive = rightward]
            Y_mm[i] =  (O.y − y_px[i]) × S       [positive = upward;
                                                    image y grows downward,
                                                    so the sign is flipped]

    ── Gap filling (preprocessing for filter) ────────────────────────────────

    4.  FOR each NaN in X_mm and Y_mm:
            LINEARLY INTERPOLATE from the nearest valid neighbours
        (Prevents the filter from collapsing or producing artefacts at gaps)

    ── Low-pass filtering ────────────────────────────────────────────────────

    5.  IF filter_method is "butterworth":

            DESIGN 4th-order Butterworth low-pass filter:
                normalised cutoff = f_c / (fps / 2)     [fraction of Nyquist]
                compute filter coefficients b, a

            APPLY zero-phase filtering (forward then reverse pass):
                X_filtered = filtfilt(b, a, X_mm)
                Y_filtered = filtfilt(b, a, Y_mm)

            (Zero-phase filtering introduces no time shift, preserving the
             phase of the pendulum oscillation)

        ELSE IF filter_method is "moving_average":

            APPLY centered moving-average over a user-set window W:
                X_filtered[i] = mean( X_mm[i − W/2 .. i + W/2] )
                Y_filtered[i] = mean( Y_mm[i − W/2 .. i + W/2] )

        ELSE (no filter):
            X_filtered = X_mm
            Y_filtered = Y_mm

    ── Build output ──────────────────────────────────────────────────────────

    6.  FOR each frame i:
            timestamp[i] = i / fps      [seconds, relative to frameStart]

    7.  OUTPUT table with columns:
            [ Frame index | Timestamp (s) | X_raw (mm) | Y_raw (mm)
              | X_filtered (mm) | Y_filtered (mm) ]

END PROCEDURE
```

---

## Summary of Key Design Decisions

| Decision | Reason |
|---|---|
| Bounding-box centre instead of contour centroid | Averages both edges of the disk; threshold jitter on one edge cancels with the other |
| Dual hue range for red | Red wraps around 0°/180° on OpenCV's hue scale; a single range would miss half the colour |
| Morphological opening before contour detection | Removes speckle noise without eroding the main blob |
| Largest-contour selection | Bob is always the dominant red object inside the ROI |
| Sign flip on Y | Image pixel rows increase downward; flipping gives a physically intuitive upward-positive axis |
| Zero-phase Butterworth filter | Removes high-frequency vibration without shifting the oscillation's phase in time |
| Linear interpolation of NaN gaps | Keeps the filter valid across frames where the bob was momentarily undetected |
