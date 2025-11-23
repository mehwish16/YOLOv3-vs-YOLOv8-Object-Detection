# YOLOv3 vs YOLOv8 Object Detection Comparison

Object detection using both YOLOv3 and YOLOv8 to compare traditional OpenCV-based and modern Ultralytics implementations.

### Detect objects in both images and video streams using Deep Learning, OpenCV, Ultralytics, and Python.

This project demonstrates how object detection performance, accuracy, and speed differ between YOLOv3 and YOLOv8. YOLOv3 is implemented using OpenCV’s DNN module, while YOLOv8 uses Ultralytics’ high-level Python API.

The models are trained on the **COCO dataset**, which contains 80 object classes, including:

* People
* Vehicles (cars, buses, bicycles, trucks, etc.)
* Household items (sofas, chairs, dining tables)
* Animals (dogs, cats, birds, horses, etc.)
* Traffic signs, sports equipment, and more

For the full COCO label list, visit: 
[COCO Class Names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)


## Model Files

* `yolo-coco` : Contains YOLOv3 configuration, weights, and COCO class names.
  * These were trained by the [Darknet Team](https://pjreddie.com/darknet/yolo/).

* `yolov8n.pt` : The pre-trained Ultralytics YOLOv8 model downloaded automatically when first run.


## YOLO Object Detection in Images

### Installation

`pip install numpy`
`pip install opencv-python`
`pip install ultralytics`

### To Run the Project

#### YOLOv3

`python image_detection/yolov3_image.py --image input/images/baggage_claim.jpg`


#### YOLOv8

`python image_detection/yolov8_image.py --image input/images/baggage_claim.jpg`

### Output Example

YOLOv3 and YOLOv8 both detect multiple objects like people, suitcases, and handbags in the same image. YOLOv8 processes faster and provides smoother bounding boxes.

## YOLO Object Detection in Video Streams

### Installation

`pip install numpy`
`pip install opencv-python`
`pip install ultralytics`

### To Run the Project

#### YOLOv3

`python video_detection/yolov3_video.py --input input/videos/airport.mp4`

#### YOLOv8

`python video_detection/yolov8_video.py --input input/videos/airport.mp4`

### Output Example

In the output video, you can see objects such as cars, people, and traffic lights being detected in real time.

YOLOv3 performs well but is slower; YOLOv8 achieves faster inference while maintaining accuracy.


## Comparison Summary

| Feature       | YOLOv3                       | YOLOv8                        |
| ------------- | ---------------------------- | ----------------------------- |
| Framework     | OpenCV DNN                   | Ultralytics (PyTorch)         |
| Model Files   | `.cfg`, `.weights`, `.names` | `.pt`                         |
| Ease of Use   | Requires manual setup        | Single-line `.predict()` call |
| Speed         | Moderate                     | Fast (optimized for GPU)      |
| Visualization | Manual (OpenCV display)      | Automatic (Ultralytics)       |


## Project Structure

ProjectRoot/
│
├── image_detection/
│   ├── yolov3_image.py
│   └── yolov8_image.py
│
├── video_detection/
│   ├── yolov3_video.py
│   └── yolov8_video.py
│
├── yolo-coco/
│   ├── coco.names
│   ├── yolov3.cfg
│   └── yolov3.weights
│
├── input/
│   ├── images/
│   └── videos/
│
└── output/
    ├── yolov3/
    └── yolov8/


## Limitations

* YOLOv3 may struggle with small or densely packed objects.
* YOLOv8 provides better precision and faster inference but requires more resources.
* For datasets with many small objects, Faster R-CNN or SSD models may perform better.
