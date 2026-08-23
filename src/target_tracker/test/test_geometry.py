import numpy as np

from target_tracker.geometry import depth_to_points
from target_tracker.geometry import optical_to_base
from target_tracker.geometry import pack_xyzi
from target_tracker.geometry import voxel_filter


def _depth(height: int, width: int, value: float) -> np.ndarray:
    return np.full((height, width), value, dtype=np.float32)


def test_centre_pixel_projects_to_the_optical_axis() -> None:
    depth = np.full((5, 5), np.nan, dtype=np.float32)
    depth[2, 2] = 2.0
    points = depth_to_points(depth, fx=100.0, fy=100.0, cx=2.0, cy=2.0, stride=1)
    assert points.shape == (1, 3)
    assert np.allclose(points[0], (0.0, 0.0, 2.0), atol=1e-6)


def test_point_ahead_becomes_x_forward_in_the_base_frame() -> None:
    # Optical (0, 0, 2) is straight down the optical axis.
    optical = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    base = optical_to_base(optical, camera_height=0.35)
    assert base[0, 0] > 0.0                       # X forward
    assert abs(float(base[0, 1])) < 1e-6          # Y centred
    assert abs(float(base[0, 2]) - 0.35) < 1e-6   # lifted by camera height


def test_depth_outside_the_band_is_dropped() -> None:
    depth = np.array([[0.5, 2.0, 9.0]], dtype=np.float32)
    points = depth_to_points(
        depth, fx=100.0, fy=100.0, cx=1.0, cy=0.0, stride=1,
        z_min=1.2, z_max=5.5,
    )
    assert points.shape == (1, 3)
    assert abs(float(points[0, 2]) - 2.0) < 1e-6


def test_stride_two_yields_a_quarter_of_the_points() -> None:
    depth = _depth(20, 20, 3.0)
    dense = depth_to_points(depth, fx=100.0, fy=100.0, cx=10.0, cy=10.0, stride=1)
    sparse = depth_to_points(depth, fx=100.0, fy=100.0, cx=10.0, cy=10.0, stride=2)
    assert dense.shape[0] == 400
    assert sparse.shape[0] == 100


def test_sparse_voxel_is_dropped_and_a_dense_one_survives() -> None:
    # Three points in one voxel, five in another 1 m away.
    thin = np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float32), (3, 1))
    thick = np.tile(np.array([[2.0, 0.0, 1.0]], dtype=np.float32), (5, 1))
    centroids, index, counts = voxel_filter(
        np.vstack((thin, thick)), resolution=0.08, min_points_per_voxel=4,
    )
    assert centroids.shape == (1, 3)
    assert index.shape == (1, 3)
    assert int(counts[0]) == 5
    assert abs(float(centroids[0, 0]) - 2.0) < 1e-5


def test_ground_and_range_gates_remove_points() -> None:
    points = np.array([
        [1.0, 0.0, 0.05],    # below ground_z
        [99.0, 0.0, 1.0],    # beyond max_range
        [1.0, 0.0, 1.0],     # keeper
    ], dtype=np.float32)
    centroids, _, counts = voxel_filter(
        points, resolution=0.08, min_points_per_voxel=1,
        ground_z=0.15, max_range=5.5,
    )
    assert centroids.shape == (1, 3)
    assert int(counts[0]) == 1


def test_empty_input_returns_empty_arrays() -> None:
    centroids, index, counts = voxel_filter(np.zeros((0, 3), dtype=np.float32))
    assert centroids.shape == (0, 3)
    assert index.shape == (0, 3)
    assert counts.shape == (0,)


def test_rejects_bad_shapes_and_intrinsics() -> None:
    try:
        depth_to_points(np.zeros(4, dtype=np.float32), fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        assert False, 'expected ValueError'
    except ValueError:
        pass
    try:
        depth_to_points(
            np.zeros((2, 2), dtype=np.float32), fx=0.0, fy=1.0, cx=0.0, cy=0.0,
        )
        assert False, 'expected ValueError'
    except ValueError:
        pass
    try:
        depth_to_points(
            np.zeros((2, 2), dtype=np.float32), fx=1.0, fy=1.0, cx=0.0, cy=0.0,
            stride=0,
        )
        assert False, 'expected ValueError'
    except ValueError:
        pass
    try:
        voxel_filter(np.zeros((4, 2), dtype=np.float32))
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_voxel_keys_match_a_row_wise_unique() -> None:
    """The packed int64 key must group exactly like np.unique(axis=0).

    voxel_filter packs each (ix, iy, iz) into one int64 because the row-wise
    unique was the most expensive step in the whole node; this pins the
    optimisation to the behaviour it replaced. The cloud deliberately straddles
    zero on all three axes: the packing shifts by the per-axis minimum, so a
    sign error there would collide two distant voxels into one key and silently
    merge two clusters.
    """
    rng = np.random.default_rng(3)
    points = np.column_stack((
        rng.uniform(-4.0, 4.0, 6000),
        rng.uniform(-4.0, 4.0, 6000),
        rng.uniform(-2.0, 2.0, 6000),
    )).astype(np.float32)
    # Gates wide open so the negative half of the cloud actually reaches the
    # packing rather than being filtered away first.
    ground_z, max_range, resolution = -10.0, 100.0, 0.25
    assert np.any(points < 0.0)

    centroids, index, counts = voxel_filter(
        points, resolution=resolution, min_points_per_voxel=1,
        ground_z=ground_z, max_range=max_range,
    )
    assert np.any(index < 0), 'expected negative voxel indices in this cloud'

    expected_index, expected_counts = np.unique(
        np.floor(points / resolution).astype(np.int32), axis=0,
        return_counts=True,
    )

    order = np.lexsort((index[:, 2], index[:, 1], index[:, 0]))
    assert np.array_equal(index[order], expected_index)
    assert np.array_equal(counts[order], expected_counts)
    assert int(counts.sum()) == int(points.shape[0])
    # No two distinct voxels may share a key, and every centroid must land
    # inside the voxel it claims to represent.
    assert len(np.unique(index, axis=0)) == index.shape[0]
    assert np.all(
        np.floor(centroids[order] / resolution).astype(np.int32) == expected_index
    )


def test_voxel_key_space_guard_rejects_an_absurd_resolution() -> None:
    """An out-of-budget key space must raise, never silently collide.

    A key collision would merge two far-apart voxels into one cluster, which is
    invisible in the output and miserable to track down later.
    """
    points = np.array([[0.0, 0.0, 1.0], [4.0, 4.0, 4.0]], dtype=np.float32)
    try:
        voxel_filter(
            points, resolution=1e-5, min_points_per_voxel=1,
            ground_z=0.0, max_range=100.0,
        )
        assert False, 'expected ValueError'
    except ValueError as exc:
        assert 'key space too large' in str(exc)
    # The same cloud at a sane resolution still works.
    centroids, _, _ = voxel_filter(
        points, resolution=0.25, min_points_per_voxel=1,
        ground_z=0.0, max_range=100.0,
    )
    assert centroids.shape[0] == 2


def test_pack_xyzi_layout_is_16_bytes_per_point() -> None:
    centroids = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    counts = np.array([7, 9], dtype=np.int64)
    blob = pack_xyzi(centroids, counts)

    assert len(blob) == 2 * 16
    back = np.frombuffer(blob, dtype=np.float32).reshape(2, 4)
    assert np.allclose(back[:, :3], centroids)
    assert np.allclose(back[:, 3], [7.0, 9.0])


def test_pack_xyzi_rejects_mismatched_inputs() -> None:
    try:
        pack_xyzi(np.zeros((3, 3), np.float32), np.zeros(2, np.int64))
        assert False, 'expected ValueError'
    except ValueError:
        pass
    try:
        pack_xyzi(np.zeros((3, 2), np.float32), np.zeros(3, np.int64))
        assert False, 'expected ValueError'
    except ValueError:
        pass
