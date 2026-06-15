"""CSV export with a physics metadata header, one file per run."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class ExportMeta:
    gap_height_mm: str
    pendulum_length_mm: float
    scale_mm_per_px: float


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "run"


def export_folder(base_dir: str, gap_height_mm: str) -> str:
    """Create and return ``<base>/height_<gap>mm/``."""
    folder = os.path.join(base_dir, f"height_{_safe(gap_height_mm)}mm")
    os.makedirs(folder, exist_ok=True)
    return folder


def run_dir(height_folder: str, run_name: str) -> str:
    """Create and return the per-run folder ``<height_folder>/<run>/``."""
    folder = os.path.join(height_folder, _safe(run_name))
    os.makedirs(folder, exist_ok=True)
    return folder


def write_run_csv(folder: str, run_name: str, df: pd.DataFrame, meta: ExportMeta) -> str:
    """Write one run's table prefixed with a commented metadata header.

    Columns: Frame, Timestamp (s), X_raw (mm), Y_raw (mm),
    X_filtered (mm), Y_filtered (mm).
    """
    path = os.path.join(folder, f"{_safe(run_name)}.csv")
    header = (
        f"# Target Gap Height (mm): {meta.gap_height_mm}\n"
        f"# Pendulum Length L (mm): {meta.pendulum_length_mm}\n"
        f"# Pixel-to-mm Scale Factor (mm/px): {meta.scale_mm_per_px:.8f}\n"
        f"# Run: {run_name}\n"
    )
    with open(path, "w", newline="") as f:
        f.write(header)
        df.to_csv(f, index=False)
    return path
