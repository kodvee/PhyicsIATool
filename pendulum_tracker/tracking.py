"""Single red-bob tracking.

The bob carries a red magnet disk; viewed from the side it reads as a short 2D
cylinder. We isolate the red region inside the ROI and take the bob position as
the **center of its bounding box** (horizontal + vertical midpoint of the red
profile). Using the centre rather than an edge averages the top/bottom (and
left/right) boundaries, so single-edge jitter from the colour threshold largely
cancels — important for the small-amplitude vertical (Y) motion, which an edge
measurement renders as a noisy staircase.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HSVRange:
    """Red hue selector. Red wraps around H=0, so we test two wedges."""

    h_center: int = 0      # 0-179 (OpenCV hue scale)
    h_tol: int = 10
    s_min: int = 90
    s_max: int = 255
    v_min: int = 70
    v_max: int = 255

    def mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo_s, hi_s = self.s_min, self.s_max
        lo_v, hi_v = self.v_min, self.v_max
        c, tol = self.h_center, self.h_tol
        # Build hue window with wrap-around handling.
        ranges = []
        low = c - tol
        high = c + tol
        if low < 0:
            ranges.append((0, high))
            ranges.append((180 + low, 179))
        elif high > 179:
            ranges.append((low, 179))
            ranges.append((0, high - 180))
        else:
            ranges.append((low, high))
        mask = None
        for h_lo, h_hi in ranges:
            m = cv2.inRange(
                hsv,
                np.array([h_lo, lo_s, lo_v], dtype=np.uint8),
                np.array([h_hi, hi_s, hi_v], dtype=np.uint8),
            )
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        # Clean speckle noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask


@dataclass
class BobResult:
    found: bool
    point: tuple[float, float] | None     # bounding-box center, full-frame px
    x_raw: float                          # = point x (full-frame px)
    y_raw: float                          # = point y (full-frame px)


def detect_bob(
    roi_bgr: np.ndarray,
    hsv: HSVRange,
    min_area: int,
    roi_offset: tuple[int, int] = (0, 0),
) -> BobResult:
    """Detect the red bob inside an ROI crop and return its bounding-box center.

    Takes the largest red region, then the centre of its bounding box:
    ``x = bbox_center_x``, ``y = bbox_center_y``. ``roi_offset`` is the (x, y) of
    the ROI's top-left corner in the full frame; the returned coordinate is
    shifted into full-frame space.
    """
    ox, oy = roi_offset
    mask = hsv.mask(roi_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not blobs:
        return BobResult(False, None, np.nan, np.nan)

    x, y, w, h = cv2.boundingRect(max(blobs, key=cv2.contourArea))
    px = x + w / 2.0 + ox       # bounding-box centre (horizontal)
    py = y + h / 2.0 + oy       # bounding-box centre (vertical)
    return BobResult(True, (px, py), px, py)


def crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``frame`` to ``roi`` = (x, y, w, h), clamped to frame bounds."""
    x, y, w, h = (int(v) for v in roi)
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return frame[y0:y1, x0:x1]
