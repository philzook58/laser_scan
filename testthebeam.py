import serial
import serial.tools.list_ports
import time
import numpy as np


import cv2
cap = cv2.VideoCapture(0)
_, frame = cap.read()

time.sleep(1)
_, nullframe = cap.read()


while True:
	_, frame = cap.read()
	subtractedframe =frame #cv2.absdiff(frame ,nullframe)
	red = subtractedframe[:,:,2]
	hotpos = np.argmax(red, axis=1)
	empty = np.zeros(frame.shape, dtype=np.uint8)
	for i in range(frame.shape[0]):
		empty[i,hotpos[i],2]=255
	#mask = np.max(red) > thresh
	cv2.imshow('gray',empty)
	print "yo"
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()





