import serial
import serial.tools.list_ports
import time
import numpy as np
port = False
baud = 57600
if port:
    port = port
else:
    port = serial.tools.list_ports.comports()[-1][0]
    #self.port = "/dev/ttyUSB0"
ser = serial.Serial(port, baud,timeout=5.)  # open serial port

angle0 = 810
angle90= 1600

startmicro = 1100 
endmicro = 1600
microSecIncrement = 1

def angle(microsec):
	microrange = angle90 - angle0
	return ((microsec-angle0)*90.)/microrange

def move(microsec):
	print str(microsec)
	ser.write(str(microsec)+'g')

import cv2
cap = cv2.VideoCapture(1)
_, frame = cap.read()

move(angle0)
time.sleep(1)
_, nullframe = cap.read()

pointcloud = []
move(startmicro)
time.sleep(1)
microsec = startmicro
while microsec < endmicro:
	_, frame = cap.read()
	microsec = microsec +  microSecIncrement
	move(microsec)
	subtractedframe = frame
	red = subtractedframe[:,:,2]
	hotpos = np.argmax(red, axis=1)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    #lower_blue = np.array([50,50,50])
    #upper_blue = np.array([70,255,255])

    #actually it is green
    lower_red = np.array([0,100,230])
    upper_red = np.array([30,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)

    
	cv2.imshow('res',cv2.pyrDown(mask))
	#mask = np.max(red) > thresh
	pointcloud.append(hotpos)

	k = cv2.waitKey(50) & 0xFF
	if k == 27:
		break


cv2.destroyAllWindows()
ser.close()
	



