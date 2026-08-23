"""Depth image → filtered 3D point cloud. Pure numpy, no ROS."""

from __future__ import annotations

import numpy as np

# Upper bound on the packed voxel key space. The key is a single int64, so an
# absurd resolution could in principle overflow it and silently merge distinct
# voxels; this turns that into an error instead. Mirrors
# target_tracker.clustering.MAX_GRID_CELLS.
MAX_VOXEL_KEYS = 1 << 40

# optical (X right, Y down, Z forward) -> REP-103 base (X forward, Y left, Z up)
R_OPTICAL_TO_BASE = np.array(
    [[0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


def depth_to_points(
    depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 3,
    z_min: float = 1.2,
    z_max: float = 5.5,
) -> np.ndarray:
    """Back-project a 32FC1 depth image into optical-frame points, shape (N, 3).

    ``stride`` subsamples the image; the intrinsics are NOT rescaled because
    the original pixel indices are preserved by the ``arange`` grids below.
    """
    if depth.ndim != 2:
        raise ValueError('expected a 2D depth image')
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError('expected positive focal lengths')
    if stride < 1:
        raise ValueError('stride must be >= 1')

    sub = np.asarray(depth[::stride, ::stride], dtype=np.float32)
    vs = np.arange(0, depth.shape[0], stride, dtype=np.float32)
    us = np.arange(0, depth.shape[1], stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    valid = np.isfinite(sub) & (sub >= z_min) & (sub <= z_max)
    z = sub[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    return np.stack((x, y, z), axis=1)


def optical_to_base(points: np.ndarray, *, camera_height: float) -> np.ndarray:
    """Rotate optical points into REP-103 and lift them by the camera height."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('expected an (N, 3) array')
    result = points @ R_OPTICAL_TO_BASE.T
    result[:, 2] += camera_height
    return result


def voxel_filter(
    points: np.ndarray,
    *,
    resolution: float = 0.08,
    min_points_per_voxel: int = 4,
    ground_z: float = 0.15,
    max_range: float = 5.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample and denoise in one pass.

    Returns ``(centroids (M, 3) float32, voxel_index (M, 3) int32,
    counts (M,) int64)``. A voxel yields one centroid only if it collected at
    least ``min_points_per_voxel`` points, which removes sparse stereo speckle.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('expected an (N, 3) array')
    if resolution <= 0.0:
        raise ValueError('resolution must be positive')

    keep = (points[:, 2] > ground_z) & (
        np.linalg.norm(points[:, :2], axis=1) < max_range
    )
    p = points[keep]
    if p.shape[0] == 0:
        return _empty_voxels()

    index = np.floor(p / resolution).astype(np.int64)
    # np.unique(..., axis=0) would lexsort the (N, 3) rows, which measured
    # ~17 ms per frame on the Pi -- the single most expensive step in the
    # pipeline. Packing each voxel into one int64 key makes it a plain 1D
    # unique over ~30k values, which is an order of magnitude cheaper.
    origin = index.min(axis=0)
    shifted = index - origin
    dims = shifted.max(axis=0) + 1
    if int(dims[0]) * int(dims[1]) * int(dims[2]) > MAX_VOXEL_KEYS:
        raise ValueError(f'voxel key space too large: {dims.tolist()}')
    keys = (shifted[:, 0] * dims[1] + shifted[:, 1]) * dims[2] + shifted[:, 2]
    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True,
    )
    inverse = np.ravel(inverse)
    unique_index = np.stack(
        np.unravel_index(unique_keys, tuple(int(d) for d in dims)), axis=1,
    ) + origin
    unique_index = unique_index.astype(np.int32)
    total = unique_index.shape[0]
    sums = np.stack([
        np.bincount(inverse, weights=p[:, axis], minlength=total)
        for axis in range(3)
    ], axis=1)
    centroids = (sums / counts[:, None]).astype(np.float32)

    good = counts >= min_points_per_voxel
    if not bool(good.any()):
        return _empty_voxels()
    return centroids[good], unique_index[good], counts[good]


def _empty_voxels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared empty return so callers can rely on the shapes and dtypes."""
    return (
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.int32),
        np.zeros(0, dtype=np.int64),
    )


def pack_xyzi(centroids: np.ndarray, counts: np.ndarray) -> bytes:
    """Pack voxel centroids into an xyz+intensity PointCloud2 payload.

    Intensity carries how many raw points the voxel swallowed, so a dense
    torso reads brighter than a one-hit speckle voxel. 16 bytes per point:
    ~600 voxels is under 10 KB per frame, which is what makes publishing the
    cloud affordable at all -- the raw depth image it came from is 1.23 MB.
    """
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError('expected an (M, 3) centroid array')
    if counts.shape[0] != centroids.shape[0]:
        raise ValueError('centroids and counts must agree in length')

    packed = np.empty((centroids.shape[0], 4), dtype=np.float32)
    packed[:, :3] = centroids
    packed[:, 3] = counts
    return packed.tobytes()
