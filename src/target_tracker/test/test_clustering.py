import numpy as np

from target_tracker.clustering import cluster_voxels
from target_tracker.geometry import voxel_filter


def _blob(center: tuple[float, float, float], *, spread: float, count: int) -> np.ndarray:
    """A small deterministic cube of points around ``center``."""
    rng = np.random.default_rng(abs(hash(center)) % (2 ** 32))
    offsets = rng.uniform(-spread, spread, size=(count, 3))
    return (np.asarray(center, dtype=np.float32) + offsets).astype(np.float32)


def _cluster(points: np.ndarray, **kwargs):
    centroids, index, counts = voxel_filter(
        points, resolution=0.08, min_points_per_voxel=1,
        ground_z=-10.0, max_range=100.0,
    )
    return cluster_voxels(centroids, index, counts, **kwargs)


def test_two_separated_blobs_give_two_clusters() -> None:
    points = np.vstack((
        _blob((1.0, 0.0, 1.0), spread=0.12, count=200),
        _blob((1.0, 1.5, 1.0), spread=0.12, count=200),
    ))
    clusters = _cluster(points, min_cluster_points=25, min_cluster_voxels=2)
    assert len(clusters) == 2


def test_one_merged_blob_gives_one_cluster() -> None:
    points = _blob((1.0, 0.0, 1.0), spread=0.25, count=400)
    clusters = _cluster(points, min_cluster_points=25, min_cluster_voxels=2)
    assert len(clusters) == 1
    assert abs(float(clusters[0].center[0]) - 1.0) < 0.1


def test_elongated_blob_has_the_larger_spread_on_x() -> None:
    rng = np.random.default_rng(7)
    points = np.column_stack((
        rng.uniform(0.5, 2.0, 600),     # 1.5 m along X
        rng.uniform(-0.1, 0.1, 600),    # 0.2 m along Y
        rng.uniform(0.9, 1.1, 600),
    )).astype(np.float32)
    clusters = _cluster(points, min_cluster_points=25, min_cluster_voxels=2)
    assert len(clusters) == 1
    assert float(clusters[0].std[0]) > float(clusters[0].std[1])
    assert float(clusters[0].size[0]) > float(clusters[0].size[1])


def test_small_blob_is_rejected_by_min_cluster_points() -> None:
    points = _blob((1.0, 0.0, 1.0), spread=0.15, count=10)
    assert _cluster(points, min_cluster_points=25, min_cluster_voxels=1) == []


def test_thin_blob_is_rejected_by_min_cluster_voxels() -> None:
    # 40 points all inside one voxel: enough points, not enough voxels.
    points = np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float32), (40, 1))
    assert _cluster(points, min_cluster_points=25, min_cluster_voxels=4) == []


def test_empty_input_returns_no_clusters() -> None:
    assert cluster_voxels(
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.int32),
        np.zeros(0, dtype=np.int64),
    ) == []


def test_size_never_collapses_to_zero() -> None:
    points = _blob((1.0, 0.0, 1.0), spread=0.02, count=100)
    clusters = _cluster(points, min_cluster_points=25, min_cluster_voxels=1)
    assert len(clusters) == 1
    assert np.all(clusters[0].size >= 0.05)


def test_rejects_mismatched_inputs() -> None:
    try:
        cluster_voxels(
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.int32),
            np.zeros(3, dtype=np.int64),
        )
        assert False, 'expected ValueError'
    except ValueError:
        pass
