# Real Time Video Streaming Guide for Raspbery Pi
This guide explains how to achieve real time video streaming from a remote IP camera connected to raspberry pi to any computer (PC/Mac/Linux). 
Streamed video from raspberry pi can also be sent directly to any OpenCV application which can be used for further processing.

### UV4L installation in Raspberry Pi

#### Pre-requisites
1. Raspberry Pi installed with raspbian operating system.
2. Camera attached to raspberry pi. Make sure you have activated the camera in raspberry pi configuration.

#### Steps to install UV4L
##### $ curl http://www.linux-projects.org/listing/uv4l_repo/lpkey.asc | sudo apt-key add -

open file /etc/apt/sources.list using nano or any preferred editor and add the below line: <br>
##### $ deb http://www.linux-projects.org/listing/uv4l_repo/raspbian/stretch stretch main <br>

After successfully saving sources.list file, execute the below commands to install uv4l: <br>
##### $ sudo apt-get update
##### $ sudo apt-get install uv4l uv4l-raspicam

UV4L script for launching, restarting, and stoping UV4L service is installed using following command: <br>
##### $ sudo apt-get install uv4l-raspicam-extras

Once the uv4l service script is installed, execute the following command to reload uv4l:<br>
##### $ sudo service uv4l_raspicam restart

Now you can view the video stream in any web browser by using the url: http://your_raspberry_pi_IP_addr:8080/stream <br>
Replace "your_raspberry_pi_addr" with the ip address of your raspberry pi. To check your ip address execute the following command:
##### $ ifconfig
If you are connected to wifi, your ip address is the inet address.

### Receive the raspberry pi video stream in OpenCV application running in computer
#### step 1
Open the OpenCV-capture-test.py file from this repository in your favorite editor.
#### Step 2
In the line 8, replace "192.168.0.186" with your raspberry pi IP address and save the file.
#### Step 3
Now execute the file using the following command to view the livestream.
##### $ python OpenCV-capture-test.py
