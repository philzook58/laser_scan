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

	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	lower_blue = np.array([0,0,230])
	upper_blue = np.array([255,255,255])
	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, lower_blue, upper_blue)

	points = np.argwhere(mask)

	points = points[:,::-1] #reverse x and y so that it is now in x,y order
	np.insert(points, 2, angle, axis=1)
	def pixelDir(x):
		# pix / f = objsize / objdist
		# f = pix * objdist / objsize
		f = 100. #camera Width of 1m object at 1m in pixels, or 8m object at 8m. 
		return np.array([ x / f ,  y / f , 1., 0.])

	cv2.imshow('res',cv2.pyrDown(mask))
	#mask = np.max(red) > thresh
	pointcloud.append(hotpos)

	k = cv2.waitKey(50) & 0xFF
	if k == 27:
		break


cv2.destroyAllWindows()
ser.close()
	



