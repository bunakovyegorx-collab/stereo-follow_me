"""Vectorized disparity → metric depth conversion (no Python pixel loops)."""

from __future__ import annotations

import numpy as np


def disparity_to_depth_m(
    disparity: np.ndarray,
    *,
    f: float,
    baseline: float,
    min_disparity: float = 0.0,
) -> np.ndarray:
    """Return axial depth Z = f·B / d as float32 metres.

    Invalid / non-positive disparity becomes NaN. Uses numpy vectorization only.
    """
    if disparity.ndim != 2:
        raise ValueError('expected a 2D disparity image')
    if f <= 0.0 or baseline <= 0.0:
        raise ValueError('expected positive focal length and baseline')

    geometry = float(f) * float(baseline)
    lower = max(0.0, float(min_disparity))
    # Promote once; keep output float32 for ROS 32FC1 bandwidth.
    d = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(d) & (d > lower)
    # Divide only where valid; elsewhere leave NaN without Python loops.
    safe = np.where(valid, d, np.nan)
    return np.asarray(geometry / safe, dtype=np.float32)
