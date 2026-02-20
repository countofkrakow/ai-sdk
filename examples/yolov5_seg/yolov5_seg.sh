#!/bin/sh

MODEL_PATH="${1:-/vendor/etc/models/yolov5_seg.nb}"
IMAGE_PATH="${2:-/vendor/etc/input_data/dog.jpg}"

# Usage:
#   sh yolov5_seg.sh [model_path] [image_path]
# Requires yolov5_seg binary with segmentation post-process support.
yolov5_seg "$MODEL_PATH" "$IMAGE_PATH"
