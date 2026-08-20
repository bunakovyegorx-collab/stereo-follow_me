import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from yolo_person_car.detector_node import Box
from yolo_person_car.detector_node import bgr_image_message
from yolo_person_car.detector_node import detection_message
from yolo_person_car.detector_node import image_to_bgr
from yolo_person_car.detector_node import validate_allowed_classes


def test_class_filter_accepts_person_and_car() -> None:
    assert validate_allowed_classes([0, 2]) == (0, 2)


def test_class_filter_rejects_unwanted_classes() -> None:
    try:
        validate_allowed_classes([0, 5])
    except ValueError as error:
        assert 'unsupported values [5]' in str(error)
    else:
        raise AssertionError('expected a ValueError')


def test_detection_message_uses_center_and_size() -> None:
    box = Box(10.0, 20.0, 30.0, 60.0, 0, 0.9)
    header = Header(frame_id='left_optical')
    header.stamp.sec = 12
    message = detection_message(box, header)

    assert message.header.frame_id == 'left_optical'
    assert message.header.stamp.sec == 12
    assert message.bbox.center.position.x == 20.0
    assert message.bbox.center.position.y == 40.0
    assert message.bbox.size_x == 20.0
    assert message.bbox.size_y == 40.0
    assert message.results[0].hypothesis.class_id == 'person'
    assert message.results[0].hypothesis.score == 0.9


def test_yuyv_input_is_converted_without_cv_bridge_encoding_support() -> None:
    # Two YUYV pixels with neutral chroma should become grayscale BGR pixels.
    message = Image(height=1, width=2, encoding='yuv422_yuy2', step=4)
    message.data = [80, 128, 180, 128]

    converted = image_to_bgr(CvBridge(), message)

    assert converted.shape == (1, 2, 3)
    assert np.allclose(converted[0, 0], [75, 75, 75], atol=2)
    assert np.allclose(converted[0, 1], [191, 191, 191], atol=2)


def test_bgr_debug_message_preserves_source_header() -> None:
    source = Image()
    source.header.frame_id = 'left_optical'
    source.header.stamp.sec = 123

    message = bgr_image_message(np.zeros((2, 3, 3), dtype=np.uint8), source)

    assert message.header.frame_id == 'left_optical'
    assert message.header.stamp.sec == 123
    assert message.encoding == 'bgr8'
    assert message.step == 9
    assert len(message.data) == 18
