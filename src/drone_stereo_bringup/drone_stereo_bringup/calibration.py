from copy import deepcopy


def scale_calibration(
    source: dict[str, object],
    width: int,
    height: int,
    camera_name: str,
    source_crop: tuple[int, int, int, int] | None = None,
) -> dict[str, object]:
    source_width = int(source['image_width'])
    source_height = int(source['image_height'])
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError('calibration dimensions must be positive')

    if source_crop is None:
        source_crop = (0, 0, source_width, source_height)
    crop_x, crop_y, crop_width, crop_height = source_crop
    if (
        min(crop_x, crop_y, crop_width, crop_height) < 0
        or crop_width == 0
        or crop_height == 0
        or crop_x + crop_width > source_width
        or crop_y + crop_height > source_height
    ):
        raise ValueError('source crop must fit inside calibration dimensions')

    result = deepcopy(source)
    scale_x = width / crop_width
    scale_y = height / crop_height

    camera_matrix = result['camera_matrix']['data']
    projection_matrix = result['projection_matrix']['data']
    if len(camera_matrix) != 9 or len(projection_matrix) != 12:
        raise ValueError('invalid camera or projection matrix size')

    camera_matrix[0] *= scale_x
    camera_matrix[2] = (camera_matrix[2] - crop_x) * scale_x
    camera_matrix[4] *= scale_y
    camera_matrix[5] = (camera_matrix[5] - crop_y) * scale_y
    projection_matrix[0] *= scale_x
    projection_matrix[2] = (projection_matrix[2] - crop_x) * scale_x
    projection_matrix[3] *= scale_x
    projection_matrix[5] *= scale_y
    projection_matrix[6] = (projection_matrix[6] - crop_y) * scale_y
    projection_matrix[7] *= scale_y

    result['image_width'] = width
    result['image_height'] = height
    result['camera_name'] = camera_name
    return result
