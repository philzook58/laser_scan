import serial
import serial.tools.list_ports
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

port = False
baud = 57600
if port:
    port = port
else:
    port = serial.tools.list_ports.comports()[-1][0]
    #self.port = "/dev/ttyUSB0"
ser = serial.Serial(port, baud,timeout=5.)  # open serial port


def angle(microsec):
	microrange = angle90 - angle0
	return ((microsec-angle0)*90.)/microrange

def move(microsec):
	print str(microsec)
	ser.write(str(microsec)+'g')

import cv2
cap = cv2.VideoCapture(1)
_, frame = cap.read()

move(1400)

time.sleep(1)

_, frame = cap.read()


hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

lower_blue = np.array([0,0,220])
upper_blue = np.array([255,255,255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask= mask)

#cv2.imshow('frame',frame)
#cv2.imshow('mask',mask)
#cv2.imshow('res',res)
cv2.imshow('res',cv2.pyrDown(mask))

#cv2.imshow("", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
indices =  np.random.randint(hsv.shape[0]*hsv.shape[1], size = 5000)
ax.scatter(hsv[:,:,0].flatten()[indices], hsv[:,:,1].flatten()[indices], hsv[:,:,2].flatten()[indices])
ax.set_xlabel('H')
ax.set_ylabel('S')
ax.set_zlabel('V')

#plt.show()





ser.close()
	



