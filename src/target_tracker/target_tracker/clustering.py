"""Voxel-grid connected components. Cheaper than DBSCAN, same intent.

The original onboard_detector runs DBSCAN (eps 0.1 m, min 10 points) over the
raw cloud. On a Raspberry Pi 5 that is far too slow, so the points are first
snapped to a voxel grid (see :mod:`target_tracker.geometry`) and neighbouring
occupied voxels are merged with 26-connectivity. With eps ~= the voxel
diagonal the grouping is equivalent, but all of the work happens inside
``scipy.ndimage.label``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# A dense bool grid is allocated over the occupied bounding box. 20M cells is
# ~20 MB, already generous for a 5.5 m box at 0.08 m; anything larger means the
# inputs are wrong (stray far-away points) and would thrash the Pi's memory.
MAX_GRID_CELLS = 20_000_000

_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


@dataclass(frozen=True)
class Cluster:
    """One detected mass of points, in the base frame."""

    center: np.ndarray      # (3,) weighted centroid
    size: np.ndarray        # (3,) bounding box extent
    std: np.ndarray         # (3,) spread of points per axis
    point_count: int


def cluster_voxels(
    centroids: np.ndarray,
    voxel_index: np.ndarray,
    counts: np.ndarray,
    *,
    min_cluster_points: int = 25,
    min_cluster_voxels: int = 4,
) -> list[Cluster]:
    """Group neighbouring voxels (26-connectivity) into clusters."""
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError('expected an (M, 3) centroid array')
    if voxel_index.shape != centroids.shape or counts.shape[0] != centroids.shape[0]:
        raise ValueError('centroids, voxel_index and counts must agree in length')
    if centroids.shape[0] == 0:
        return []

    origin = voxel_index.min(axis=0)
    shape = (voxel_index.max(axis=0) - origin + 1).astype(np.int64)
    if int(np.prod(shape)) > MAX_GRID_CELLS:
        raise ValueError(f'voxel grid too large: {shape.tolist()}')

    local = (voxel_index - origin).astype(np.int64)
    grid = np.zeros(tuple(shape), dtype=bool)
    grid[local[:, 0], local[:, 1], local[:, 2]] = True

    labels, count = ndimage.label(grid, structure=_CONNECTIVITY_26)
    if count == 0:
        return []
    voxel_labels = labels[local[:, 0], local[:, 1], local[:, 2]]

    clusters: list[Cluster] = []
    for label in range(1, count + 1):
        member = voxel_labels == label
        if int(member.sum()) < min_cluster_voxels:
            continue
        points = centroids[member]
        # Weight each voxel by how many raw points it swallowed, so a dense
        # torso pulls the centre harder than a one-hit speckle voxel.
        weights = counts[member].astype(np.float64)
        total = float(weights.sum())
        if total < min_cluster_points:
            continue

        center = (points * weights[:, None]).sum(axis=0) / total
        deviation = points - center
        variance = (deviation ** 2 * weights[:, None]).sum(axis=0) / total
        # Floor the extent: a single-voxel-thick slab would otherwise get a
        # zero-width box that no 3D viewer can draw.
        size = np.maximum(points.max(axis=0) - points.min(axis=0), 0.05)

        clusters.append(Cluster(
            center=center.astype(np.float32),
            size=size.astype(np.float32),
            std=np.sqrt(variance).astype(np.float32),
            point_count=int(total),
        ))
    return clusters
