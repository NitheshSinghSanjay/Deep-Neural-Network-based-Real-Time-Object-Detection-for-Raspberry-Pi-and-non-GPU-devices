#This program is for testing the OpenCV videocapture
#to receive video stream from the local webcam and also from
#the ip raspberry pi camera 
import numpy as np
import cv2

#Comment the line 4 and uncomment the line 5 to switch to local webcam
cap = cv2.VideoCapture('http://192.168.0.186:8080/stream/video.mjpeg')
#cap = cv2.videoCapture(0);

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',grey)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
	
