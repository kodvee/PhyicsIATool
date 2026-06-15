"""Low-pass filtering to strip the hook's secondary high-frequency yaw.

The bob translation is a slow exponential-decay swing (< ~2 Hz); the hook adds
an independent high-frequency vibration. A zero-phase low-pass with a 1-5 Hz
cutoff removes the vibration while preserving the swing envelope and phase.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def _interpolate_nans(y: np.ndarray) -> np.ndarray:
    """Linearly fill NaN gaps (dropped detections) so filtering stays valid."""
    y = np.asarray(y, dtype=float).copy()
    nans = np.isnan(y)
    if nans.all():
        return y
    if nans.any():
        idx = np.arange(len(y))
        y[nans] = np.interp(idx[nans], idx[~nans], y[~nans])
    return y


def butterworth_lowpass(signal, cutoff_hz: float, fs: float, order: int = 4):
    """Zero-phase Butterworth low-pass. Falls back gracefully on short signals."""
    y = _interpolate_nans(signal)
    n = len(y)
    nyq = fs / 2.0
    if n < 9 or cutoff_hz <= 0 or cutoff_hz >= nyq:
        return y  # nothing sensible to do
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    padlen = 3 * max(len(a), len(b))
    if n <= padlen:
        return y
    return filtfilt(b, a, y)


def moving_average(signal, window: int):
    """Centered moving-average smoother (odd window enforced)."""
    y = _interpolate_nans(signal)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")
