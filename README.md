# upo_nhoa_perception

ROS package containing nodes for robotic perception, developed at UPO.

## Dependencies

You need to install [`upo_nhoa_msgs`](https://github.com/nhoa-project/upo_nhoa_msgs) and ONNX Runtime:

```bash
wget https://robotics.upo.es/~famozur/onnx/onnxruntime-gpu_1.14.1_amd64.deb
sudo apt install ./onnxruntime-gpu_1.14.1_amd64.deb
```

## YOLOv8 object detection

This is an object detection node based on [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics). It subscribes to an RGB camera topic, and publishes topics with the object detection result. If no nodes are subscribed to any of the output topics, no detection process is performed and thus no computational resources are used.

```bash
roslaunch upo_nhoa_perception yolov8.launch [args...]
```

Input topics:
- RGB camera (see `image_topic` and `image_transport` arguments)

Output topics:
- List of detected objects (`upo_nhoa_msgs/DetectionList2D`), at `/$(node_name)/objects`
- Image with bounding boxes overlaid for debugging (`sensor_msgs/Image`), at `/$(node_name)/debug_image`

Launch file arguments:
- `node_name` (Default: `yolov8`): Specifies the name of the node.
- `image_topic` (Default: `/xtion/rgb/image_raw`): Specifies the input topic for the RGB camera.
- `image_transport` (Default: `compressed`): Specifies the transport type for the image topic.
- `data_dir` (Default: `<package-dir>/data`): Specifies the path to the folder containing model files.
- `cuda_dir` (Default: `/usr/local/cuda/lib64`): Specifies the path to the necessary CUDA Runtime libraries.
- `model_file` (Default: `yolov8m.onnx`): Specifies the filename of the pre-converted YOLOv8 model (relative to `data_dir`).
- `obj_names_file`: Specifies the filename of the object name database (relative to `data_dir`).
- `obj_score_thresh` (Default: `0.1`): Specifies the minimum score that will be considered as an object.
- `nms_thresh` (Default: `0.5`): Specifies the minimum IoU score that will be used during Non-Maximum Suppression of object proposals.

In order to convert a PyTorch YOLOv8 model to ONNX you can use the following command:

```bash
yolo export model=yolov8m.pt format=onnx
```
