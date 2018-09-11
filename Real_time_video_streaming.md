# Real Time Video Streaming Guide for Raspbery Pi
This guide explains how to achieve real time video streaming from a remote IP camera connected to raspberry pi to any computer (PC/Mac/Linux). 
Streamed video from raspberry pi can also be sent directly to any OpenCV application which can be used for further processing.

### UV4L installation in Raspberry Pi

$ curl http://www.linux-projects.org/listing/uv4l_repo/lpkey.asc | sudo apt-key add -

open file /etc/apt/sources.list using nano or any preferred editor and add the below line: <br>
$ deb http://www.linux-projects.org/listing/uv4l_repo/raspbian/stretch stretch main <br>

After successfully saving sources.list file, execute the below commands to install uv4l: <br>
$ sudo apt-get update <br>
$ sudo apt-get install uv4l uv4l-raspicam

UV4L service script for launching, restarting, and stoping UV4L is installed using following command: <br>
$ sudo apt-get install uv4l-raspicam-extras <br>

Once the uv4l service script is installed, execute the following command to reload uv4l:<br>
$ sudo service uv4l_raspicam restart
