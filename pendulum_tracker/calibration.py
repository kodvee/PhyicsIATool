"""Spatial and coordinate-system calibration.

Scale: two clicked points + a typed physical distance give a pixel->mm factor.
Origin: the resting marker position. The Y-axis is a true vertical line through
the origin (image columns are assumed perpendicular to the level camera sensor),
and Y is reported positive-upward (image y grows downward, hence the sign flip).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Calibration:
    scale_mm_per_px: float          # S
    origin_px: tuple[float, float]  # (x, y) in full-frame pixels

    def to_mm(self, x_px, y_px):
        """Convert full-frame pixel coords to mm in the origin-centered frame.

        Accepts scalars or numpy arrays. X is positive-right, Y positive-up.
        """
        ox, oy = self.origin_px
        s = self.scale_mm_per_px
        x_mm = (np.asarray(x_px, dtype=float) - ox) * s
        y_mm = (oy - np.asarray(y_px, dtype=float)) * s  # flip to up-positive
        return x_mm, y_mm


def scale_from_points(p1, p2, physical_distance_mm: float) -> float:
    """pixel->mm scale factor S = physical_distance / pixel_distance."""
    px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if px <= 0:
        raise ValueError("The two scale points are identical (zero pixel distance).")
    return physical_distance_mm / px
