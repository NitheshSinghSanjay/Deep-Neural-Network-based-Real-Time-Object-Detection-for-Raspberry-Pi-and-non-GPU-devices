import time
import cv2

cap = cv2.VideoCapture(0)

while(True):
	start_time = time.time()
	ret, frame = cap.read()
	for x in range(0, 200):
		sum1 = 0.0 + 1.0**2 / 1.0**2  # sum1 == 1.0
		sum3 = 0.0 + 1.0 / 1.0**2     # sum3 == 1.0
		sum5 = 0.0 + 1 / 1.0**2
		sum6 = 0.0 + 1.0**2 / 1.0**2  # sum1 == 1.0
		sum7 = 0.0 + 1.0 / 1.0**2     # sum3 == 1.0
		sum8 = 0.0 + 1 / 1.0**2
	print("FPS: ", 1.0/ (time.time() - start_time))
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()