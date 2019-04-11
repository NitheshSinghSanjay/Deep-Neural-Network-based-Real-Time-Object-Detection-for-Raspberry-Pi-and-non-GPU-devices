# Design and Implementation of a Deep Neural Network based Real Time Object Detection for Raspberry Pi and non-GPU Devices.
## Abstract
In this project we present a new neural network architecture, MobileNet-Tiny that can be used to harness the power of GPU based real-time object detection in raspberry-pi and also in devices with the absence of a GPU / restricted graphic processing capabilities such as mobile phones, laptops, etc. MobileNet-Tiny trained on COCO dataset running on a non-Gpu laptop dell xps 13, achieves an accuracy of 19.0 mAP and a speed of 19.4 FPS which is 3 times as fast as MobileNetV2, and when running on a raspberrypi it achieves a speed of 4.5 FPS which is up to 7 times faster than MobileNetV2. MobileNet-Tiny was modeled to offer a compact, quick and well balanced object detection solution to variety of GPU restricted devices.

## Introduction
Internet of things(IoT) has shown great proliferation and has had an immense effect on our daily lives since its inception in 1999 in various ways such as advancements in security systems, home automation, and health monitoring systems etc. According to the authors of "Research on the architecture and key technology of internet of things (iot) applied on smart grid,", one of the three important characteristics of IoT is its Sensing capability with the help of sensors, cameras, RFID, GPS and other data collection devices. Information from these sensors is collected using a reliable communication network. While most of the collected information can be processed at the edge, some applications that compel object detection capabilities require more powerful base stations or cloud to process the image/video information sent by the edge and detect objects in real-time. Currently, state-of-the-art object detection algorithms used for object detection require very high GPU computing power and even then, are not always able to detect in real-time.

MobileNet-Tiny tries to address this problem. Using the Single Shot Multibox Detector (SSD) based MobileNetV2 as a starting point, MobileNet-Tiny is an attempt to get a real time object detection algorithm on non-GPU computers and edge device such as Raspberry Pi. This can offload huge computation overhead off the base station or even eliminate the requirement of the base station for real-time object detection.

## Goal
The goal with MobileNet-Tiny is to create a model that can run at a minimum of 3 frames per second (FPS) on a Raspberry Pi (GoPiGo) without the requirement of a base station or a cloud for object detection with a mAP of ~22.1% as achieved by original MobileNetV2 on MS COCO dataset.

## Contribution
MobileNet-Tiny offers three main contributions to the field of object detection. This model:
* Demonstrates the power of small artificial neural networks with fast non-GPU object detection capabilities.
* Suggests that since batch normalization in small neural networks increase the total number of parameters and total number of computations required and contribute very less in achieving high accuracy, not implementing batch normalization in small networks can drastically increase the overall speed of the network without signifcant loss of accuracy. 
* Suggests that carefully optimizing the number of predictor layers and aspect ratios for anchor boxes in SSD for small networks can result in significant improvement of detection speed.
