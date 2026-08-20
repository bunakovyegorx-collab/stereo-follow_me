# YOLO person/car detector

This package is an independent ROS 2 detector for COCO `person` and `car`.
It uses an explicitly exported YOLO26n NCNN model and does not start the
stereo depth pipeline.

## One-time setup

```bash
sudo apt install ros-jazzy-vision-msgs python3-venv
sudo apt install --no-install-recommends python3-torch python3-torchvision
python3 -m venv --system-site-packages ~/.venvs/camera-yolo
~/.venvs/camera-yolo/bin/pip install ncnn==1.0.20260526
~/.venvs/camera-yolo/bin/pip install --no-deps ultralytics==8.4.94
~/.venvs/camera-yolo/bin/pip install --no-deps matplotlib==3.11.0 polars==1.42.1 ultralytics-thop==2.0.20 nvidia-ml-py==13.610.43
~/.venvs/camera-yolo/bin/pip install pnnx==20260526
mkdir -p ~/.local/share/camera-yolo/models
cd ~/.local/share/camera-yolo/models
~/.venvs/camera-yolo/bin/yolo export model=yolo26n.pt format=ncnn imgsz=320
```

The export command is the explicit model download and conversion step. It must
finish with `yolo26n_ncnn_model/` in the models directory.

## Run the person test

```bash
source /opt/ros/jazzy/setup.bash
source ~/camera_ws/install/setup.bash
ros2 launch yolo_person_car person_test.launch.py
ros2 run rqt_image_view rqt_image_view /yolo_test/debug_image
ros2 topic echo /yolo_test/detections --once --qos-reliability best_effort
```

The detector subscribes to `image_raw` in its namespace and publishes
`/yolo_test/detections` (`vision_msgs/Detection2DArray`) and
`/yolo_test/debug_image` (`sensor_msgs/Image`). Both preserve the source image
timestamp and frame ID. The node keeps only the newest available frame.

## Parameters

- `image_topic` (default `image_raw`): input image topic.
- `model_path`: NCNN model directory; no default download occurs at startup.
- `confidence_threshold` (default `0.40`) and `iou_threshold` (default `0.45`).
- `imgsz` (default `320`): YOLO inference size.
- `allowed_classes` (default `[0, 2]`): only COCO person and car are accepted.
- `enable_debug_image` (default `true`): publish annotated BGR images.
- `inference_period_sec` (default `0.01`): polling period for the newest frame.
- `max_inference_fps` (default `0`, unlimited): cap inference while retaining
  only the newest frame; the combined person-range pipeline uses `3.0`.
