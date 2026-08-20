"""Build a stereo_msgs/DisparityImage from StereoNet's raw output tensor.

The field conventions here match the standard message definition exactly
(min_disparity/max_disparity as absolute bounds, t as the baseline in
metres) and the decode logic in
drone_stereo_bringup/disparity_viz_node.py, so that node's existing
32FC1-to-mono8 converter works unchanged against this publisher via a
simple topic remap.

Honesty note: StereoNet's raw output is NOT a calibrated pixel-disparity
value -- Hailo's own reference C++ example just visualizes it as 8-bit
grayscale. This module does not invent a calibrated scale; callers are
expected to supply min_disparity/max_disparity derived from the observed
output range (see stereonet_node.py), not from an assumed calibration.
"""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from stereo_msgs.msg import DisparityImage


def build_disparity_message(
    raw: np.ndarray,
    header: Header,
    f: float,
    baseline_m: float,
    min_disparity: float,
    max_disparity: float,
) -> DisparityImage:
    """Wrap a dense per-pixel tensor into a DisparityImage message.

    Args:
        raw: 2D (or squeezable-to-2D) array of per-pixel raw model output.
        header: header to stamp on both the outer message and the image.
        f: focal length in pixels (already scaled for any resize applied).
        baseline_m: stereo baseline in metres (``t`` field).
        min_disparity: absolute lower bound; pixels below this are invalid.
        max_disparity: absolute upper bound; must exceed min_disparity.
    """
    if max_disparity <= min_disparity:
        raise ValueError('max_disparity must exceed min_disparity')

    raw = np.squeeze(np.asarray(raw, dtype=np.float32))
    if raw.ndim != 2:
        raise ValueError('expected a 2D disparity array after squeeze')

    # Sentinel below min_disparity marks non-finite pixels as invalid,
    # matching disparity_viz_node's `disparity >= minimum` validity check.
    raw = np.nan_to_num(
        raw, nan=min_disparity - 1.0, posinf=max_disparity, neginf=min_disparity - 1.0,
    )

    image = Image()
    image.header = header
    image.height, image.width = raw.shape
    image.encoding = '32FC1'
    image.is_bigendian = False
    image.step = image.width * 4
    image.data = raw.tobytes()

    message = DisparityImage()
    message.header = header
    message.image = image
    message.f = float(f)
    message.t = float(baseline_m)
    message.min_disparity = float(min_disparity)
    message.max_disparity = float(max_disparity)
    # Unknown quantization step for an uncalibrated raw output -- do not
    # claim a precision figure that hasn't been measured.
    message.delta_d = 0.0
    return message
