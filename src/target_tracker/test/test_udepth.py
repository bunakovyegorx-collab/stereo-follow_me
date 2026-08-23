import numpy as np

from target_tracker.udepth import build_u_map
from target_tracker.udepth import extract_u_boxes
from target_tracker.udepth import u_box_to_3d

Z_MIN, Z_MAX, BINS, COL_SCALE = 1.2, 5.5, 64, 0.5


def _blank(height: int = 120, width: int = 160) -> np.ndarray:
    return np.full((height, width), np.nan, dtype=np.float32)


def _u_map(depth: np.ndarray, *, blur: bool = False) -> np.ndarray:
    return build_u_map(
        depth, z_min=Z_MIN, z_max=Z_MAX, bins=BINS,
        col_scale=COL_SCALE, blur=blur,
    )


def test_u_map_shape_and_bin_placement() -> None:
    depth = _blank()
    depth[:, :] = 3.0
    u_map = _u_map(depth)
    assert u_map.shape == (BINS, 80)
    # 3.0 m lands in bin floor((3.0 - 1.2) / 4.3 * 64) = 26.
    row = int((3.0 - Z_MIN) / (Z_MAX - Z_MIN) * BINS)
    assert int(u_map[row].max()) == 120
    assert int(u_map.sum()) == 120 * 80


def test_solid_wall_gives_one_full_width_box() -> None:
    depth = _blank()
    depth[:, :] = 3.0
    boxes = extract_u_boxes(_u_map(depth), min_count=6)
    assert len(boxes) == 1
    x, _, w, _ = boxes[0]
    assert x == 0
    assert w == 80


def test_two_people_at_different_depths_give_two_boxes() -> None:
    depth = _blank()
    depth[20:100, 10:40] = 2.0     # left, near
    depth[20:100, 110:140] = 4.0   # right, far
    boxes = sorted(extract_u_boxes(_u_map(depth), min_count=6))
    assert len(boxes) == 2
    left, right = boxes
    assert left[0] < right[0]      # different columns
    assert left[1] != right[1]     # different depth bins
    assert left[1] < right[1]      # the nearer one sits higher in the U-map


def test_empty_depth_yields_no_boxes() -> None:
    assert extract_u_boxes(_u_map(_blank()), min_count=6) == []


def test_floor_smears_across_bins_and_fails_the_threshold() -> None:
    # A ramp: every row of a column is at a different depth, so no bin
    # accumulates a tall count -- exactly how the ground plane behaves.
    depth = _blank(height=120, width=160)
    ramp = np.linspace(Z_MIN + 0.01, Z_MAX - 0.01, 120, dtype=np.float32)
    depth[:, :] = ramp[:, None]
    u_map = _u_map(depth)
    assert int(u_map.max()) <= 3


def test_u_box_to_3d_recovers_the_wall_distance() -> None:
    depth = _blank()
    depth[:, :] = 3.0
    boxes = extract_u_boxes(_u_map(depth), min_count=6)
    box = u_box_to_3d(
        depth, boxes[0], fx=200.0, fy=200.0, cx=80.0, cy=60.0,
        z_min=Z_MIN, z_max=Z_MAX, bins=BINS, col_scale=COL_SCALE,
    )
    assert box is not None
    assert abs(float(box.center[2]) - 3.0) < 0.2


def test_u_box_to_3d_recovers_a_person_height() -> None:
    depth = _blank(height=200, width=160)
    # 100 px tall at 3 m with fy=200 -> 100 * 3 / 200 = 1.5 m.
    depth[50:150, 60:90] = 3.0
    boxes = extract_u_boxes(_u_map(depth), min_count=6)
    assert len(boxes) == 1
    box = u_box_to_3d(
        depth, boxes[0], fx=200.0, fy=200.0, cx=80.0, cy=100.0,
        z_min=Z_MIN, z_max=Z_MAX, bins=BINS, col_scale=COL_SCALE,
    )
    assert box is not None
    assert abs(float(box.size[1]) - 1.5) < 0.25


def test_u_box_to_3d_returns_none_when_the_run_is_too_short() -> None:
    depth = _blank()
    depth[10:14, 20:60] = 3.0     # only 4 rows tall
    boxes = extract_u_boxes(_u_map(depth), min_count=3)
    assert boxes
    assert u_box_to_3d(
        depth, boxes[0], fx=200.0, fy=200.0, cx=80.0, cy=60.0,
        z_min=Z_MIN, z_max=Z_MAX, bins=BINS, col_scale=COL_SCALE,
        min_run=8,
    ) is None


def test_blur_merges_neighbouring_depth_bins() -> None:
    depth = _blank()
    depth[0:60, 20:60] = 3.00
    depth[60:120, 20:60] = 3.10   # one bin away
    sharp = extract_u_boxes(_u_map(depth, blur=False), min_count=6)
    blurred = extract_u_boxes(_u_map(depth, blur=True), min_count=6)
    assert len(blurred) <= len(sharp)
    assert len(blurred) == 1


def test_rejects_bad_inputs() -> None:
    for kwargs in (
        {'bins': 2},
        {'col_scale': 0.0},
        {'col_scale': 1.5},
    ):
        try:
            build_u_map(np.zeros((4, 4), dtype=np.float32), **kwargs)
            assert False, f'expected ValueError for {kwargs}'
        except ValueError:
            pass
    try:
        build_u_map(np.zeros(4, dtype=np.float32))
        assert False, 'expected ValueError'
    except ValueError:
        pass
    try:
        extract_u_boxes(np.zeros(4, dtype=np.uint16))
        assert False, 'expected ValueError'
    except ValueError:
        pass
