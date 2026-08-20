#!/usr/bin/env bash
set +u
LOG="$HOME/camera_ws/stereo_calibrator.log"
{
  echo "=== stereo calibrator start $(date -Is) ==="
  export DISPLAY=:0
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
  export WAYLAND_DISPLAY=wayland-0
  source /opt/ros/jazzy/setup.bash
  source ~/camera_ws/install/setup.bash
  exec ros2 run camera_calibration cameracalibrator --approximate 0.1 \
    --size 8x6 --square 0.030 --ros-args \
    -r left:=/stereo/left/image_raw \
    -r right:=/stereo/right/image_raw \
    -r left_camera/set_camera_info:=/stereo/left/camera_node/set_camera_info \
    -r right_camera/set_camera_info:=/stereo/right/camera_node/set_camera_info
} >> "$LOG" 2>&1
