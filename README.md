# SSD-based-object-detection-and-collision-avoidance-for-GoPiGo
## Abstract
Internet of things has shown great proliferation and has had an immense effect
on our daily lives since its inception in 1999 in various ways such as advancements in security systems, home automation, and health monitoring systems etc. One of the three important characteristics of internet of things is its Sensing capability with the help of sensors, cameras, RFID, GPS and other data collection devices. Information from these sensors are collected using a using a reliable communication network. While most of the collected information can be processed in the edge, some applications that compel object detection capabilities require more powerful base stations or cloud to process the image/video information sent by the edge and detect objects in real-time. Currently, state-of-the-art object detection algorithms used for object detection require very high GPU computing
power and even then, are not always able to detect in real-time.

## Solution
MobileNet-Tiny tries to address this problem. Using the Single Shot Multibox Detector (SSD) based MobileNetV2 as a starting point, MobileNet-Tiny is an attempt to get a real time object detection algorithm on non-GPU computers and edge device such as Raspberry Pi. This can offoad huge computation overhead of the base station or even eliminate the requirement of the base station for real-time object detection.

## Goal
The goal with MobileNet-Tiny is to develop an architecture that can run at a minimum of 3 frames per second (FPS) on a Raspberry Pi (GoPiGo) without the requirement of a base station or a cloud for object detection with a mAP of ~22.1% on MS COCO dataset and ~33% on PASCAL VOC dataset.

### Contribution
MobileNet-Tiny offers three main contributions to the field of object detection. This model:
* Demonstrates the power of small articial neural networks with fast non-GPU object detection capabilities.
* Suggests that since batch normalization in small neural networks increase the total number of parameters and total number of computations required and contribute very less in achieving high accuracy, not implementing batch normalization in small networks can drastically increase the overall speed of the network without signicant loss of accuracy. 
* Suggests that carefully optimizing the number of predictor layers and aspect ratios for anchor boxes in SSD can result in signicant improvement of detection speed.

This project is still in progress
https://www.linux-projects.org/uv4l/installation/
