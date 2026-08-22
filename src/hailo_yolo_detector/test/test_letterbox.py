import numpy as np

from hailo_yolo_detector.letterbox import letterbox
from hailo_yolo_detector.letterbox import undo_letterbox_box


def test_letterbox_square_input_has_no_padding() -> None:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    canvas, transform = letterbox(image, 640)

    assert canvas.shape == (640, 640, 3)
    assert transform.scale == 1.0
    assert transform.pad_x == 0.0
    assert transform.pad_y == 0.0


def test_letterbox_wide_image_pads_top_and_bottom() -> None:
    image = np.zeros((240, 640, 3), dtype=np.uint8)
    canvas, transform = letterbox(image, 640)

    assert canvas.shape == (640, 640, 3)
    assert transform.pad_x == 0.0
    assert transform.pad_y > 0.0
    # Padding should be gray (114, 114, 114) at the top edge.
    assert tuple(int(v) for v in canvas[0, 0]) == (114, 114, 114)


def test_letterbox_tall_image_pads_left_and_right() -> None:
    image = np.zeros((640, 240, 3), dtype=np.uint8)
    canvas, transform = letterbox(image, 640)

    assert canvas.shape == (640, 640, 3)
    assert transform.pad_y == 0.0
    assert transform.pad_x > 0.0


def test_undo_letterbox_box_round_trips_for_square_input() -> None:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    _, transform = letterbox(image, 640)

    sx1, sy1, sx2, sy2 = undo_letterbox_box(100.0, 50.0, 300.0, 400.0, transform)

    assert (sx1, sy1, sx2, sy2) == (100.0, 50.0, 300.0, 400.0)


def test_undo_letterbox_box_accounts_for_padding_and_scale() -> None:
    # 320x240 source scaled up to fit 640x640 -> scale=2.0, pad_y=80, pad_x=0.
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    _, transform = letterbox(image, 640)

    assert transform.scale == 2.0
    assert transform.pad_y == 80.0

    sx1, sy1, sx2, sy2 = undo_letterbox_box(0.0, 80.0, 640.0, 560.0, transform)

    assert (sx1, sy1, sx2, sy2) == (0.0, 0.0, 320.0, 240.0)


def test_undo_letterbox_box_clamps_to_source_bounds() -> None:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    _, transform = letterbox(image, 640)

    sx1, sy1, sx2, sy2 = undo_letterbox_box(-50.0, -50.0, 700.0, 700.0, transform)

    assert (sx1, sy1, sx2, sy2) == (0.0, 0.0, 640.0, 640.0)
