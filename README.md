# Design and Implementation of a Deep Neural Network based Real Time Object Detection for Raspberry Pi and non-GPU Devices.
## Abstract
In this project we present a new neural network architecture, MobileNet-Tiny that can be used to harness the power of GPU based real-time object detection in raspberry-pi and also in devices with the absence of a GPU / restricted graphic processing capabilities such as mobile phones, laptops, etc. MobileNet-Tiny trained on COCO dataset running on a non-Gpu laptop dell xps 13, achieves an accuracy of 19.0 mAP and a speed of 19.4 FPS which is 3 times as fast as MobileNetV2, and when running on a raspberrypi it achieves a speed of 4.5 FPS which is up to 7 times faster than MobileNetV2. MobileNet-Tiny was modeled to offer a compact, quick and well balanced object detection solution to variety of GPU restricted devices.

## Results
You can see results and example detections here: https://nitheshsinghsanjay.github.io/
link to download the paper will be made available once it is published. contact us if you have any questions.

## How to use the code
Above code is for demonstration purpose only. It is trained on udacity self driving dataset which contains 5 different objects: car, truck, pedestrian, bicyclist, light.

### How to run on non-GPU laptop? 
open visualize_pb.ipynb in jupyter notebook and load the model's inference graph "mnet_fast_inference.pb" which is available in model folder and change opencv videocapture path from 'examples/sample1.mp4' to any video file or webcam(0) in your computer. run each code block in the notebook.

### How to run on raspberry pi?
1. Install opencv dnn library for raspberry pi.
2. Open raspi_test_model.py and change the path names for .pb inference graph and the input image. 
3. Run python raspi_test_model.py to visualize the result.
(You can also load video using opencv video capture)
