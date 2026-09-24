"""Helpers for the TCP pose layout stored by the original RISE dataset."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


RISE_TCP_LAYOUT = "[x,y,z,qw,qx,qy,qz]"


def parse_rise_tcp_pose(
    raw: Any,
    *,
    dtype: Any = np.float32,
    source: Optional[Any] = None,
    quaternion_norm_tolerance: float = 1e-3,
) -> np.ndarray:
    """Return an original-RISE TCP pose in PyTorch3D scalar-first layout."""
    values = np.asarray(raw).reshape(-1)
    label = f" in {source}" if source is not None else ""
    if values.size < 7:
        raise ValueError(
            f"Expected original RISE TCP {RISE_TCP_LAYOUT}{label}, got shape {values.shape}"
        )

    pose = values[:7].astype(dtype, copy=True)
    if not np.isfinite(pose).all():
        raise ValueError(f"Non-finite original RISE TCP pose{label}: {pose}")

    quat_norm = float(np.linalg.norm(pose[3:7]))
    if abs(quat_norm - 1.0) > float(quaternion_norm_tolerance):
        raise ValueError(
            f"Invalid original RISE wxyz quaternion norm{label}: "
            f"norm={quat_norm:.8f}, pose={pose}"
        )
    return pose
