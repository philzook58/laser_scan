import serial
import serial.tools.list_ports
import time
import numpy as np
import cv2

port = False
baud = 57600

angle0 = 810
angle90= 1600

PLaser = np.array([0.35,0,0])
focal = 860. 
PCameraHomog = np.array([0.,0.,0.,1.])

startmicro = 1100 
endmicro = 1600
microSecIncrement = 1

filename = "cloudtest2.xyz"


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


cap = cv2.VideoCapture(0)
_, frame = cap.read()

height, width, _ = frame.shape
print width
print height
move(angle0)
time.sleep(1)
_, nullframe = cap.read()

pointcloud = None
move(startmicro)
time.sleep(1)
microsec = startmicro

framenum = 0
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
	currentangle = angle(microsec)
	#points = np.insert(points, 2, currentangle, axis=1) #on first axis at position 2 insert the current angle
	points = np.insert(points, 2, framenum, axis=1)
	framenum += 1
	laserRadian = currentangle * np.pi /180
	laserDirHomog =  np.array([[-1.* np.cos(laserRadian), 0, np.sin(laserRadian),0]])


	cv2.imshow('res',cv2.pyrDown(mask))
	#mask = np.max(red) > thresh
	#pointcloud.append(points)
	if pointcloud == None:
		pointcloud = points
		laserDirHomogs = laserDirHomog
	else:
		pointcloud = np.concatenate((pointcloud,points), axis=0)
		laserDirHomogs = np.concatenate((laserDirHomogs,laserDirHomog), axis=0)

	k = cv2.waitKey(50) & 0xFF
	if k == 27:
		break


cv2.destroyAllWindows()
ser.close()

#print pointcloud
laserRadian = pointcloud[:,2] * np.pi / 180


PLaserHomog = np.append(PLaser, [1.])
upDirHomog = np.array([0.,1.,0.,0.])
#laserDirHomog = np.array([-1.* np.cos(laserRadian), np.sin(laserRadian), 0., 0.])
zeros = np.zeros_like(laserRadian)
ones = np.ones_like(laserRadian)
#laserDirHomog = np.array([-1.* np.cos(laserRadian), zeros, np.sin(laserRadian),zeros]).T

#print pointcloud
#print laserDirHomogs
def planeMat(laserDirHomog):

 	return np.stack((PLaserHomog, upDirHomog, laserDirHomog))

def colminor(mat,j):
	subMat = np.delete(mat, j, axis=1)
	return (-1.)**j * np.linalg.det(subMat)

#The homogenous vector describing the plane coming off of the line laser. p dot x = 0 if x is on plane

#laserPlaneHomog = np.array(map(lambda j: colminor(planeMat, j) , range(4)))
print "Calculating Laser Planes"
laserPlaneHomogs = map(lambda laserDirH: np.asarray(map(lambda j: colminor(planeMat(laserDirH), j) , range(4))) , laserDirHomogs)
laserPlaneHomogs = np.asarray(laserPlaneHomogs)

#print laserPlaneHomogs




def pixelDir(pointcloud):
	# pix / f = objsize / objdist
	# f = pix * objdist / objsize
	f = focal #100. #camera Width of 1m object at 1m in pixels, or 8m object at 8m. 
	return np.array([(pointcloud[:,0]-width/2)/f,(pointcloud[:,1]-height/2)/f,ones,zeros]).T #I could have replaced the ones with f.

pixelDirs = pixelDir(pointcloud)
#print pixelDirs


#posHomog = map(lambda laserPlaneHomog: map(lambda cameraRay: np.dot(cameraRay, laserPlaneHomog) * PCameraHomog - np.dot(PCameraHomog, laserPlaneHomog) * cameraRay, pixelDirs), laserPlaneHomogs)
posHomog = np.zeros_like(pixelDirs)
'''
print pointcloud[-1,:]
print pixelDirs[-1,:]
print laserPlaneHomogs[-1,:]
print PCameraHomog
print np.dot(pixelDirs[-1,:], laserPlaneHomogs[-1,:]) * PCameraHomog - np.dot(PCameraHomog, laserPlaneHomogs[-1,:]) * pixelDirs[-1,:]
'''
print "Calculating points"
for i in range(pixelDirs.shape[0]):
	#print pointcloud[i,2]
	posHomog[i,:] = np.dot(pixelDirs[i,:], laserPlaneHomogs[pointcloud[i,2],:]) * PCameraHomog - np.dot(PCameraHomog, laserPlaneHomogs[pointcloud[i,2],:]) * pixelDirs[i,:]
#print posHomog
#posHomog = np.asarray(posHomog)

def removeHomog(posHomog):
	return np.asarray( map(lambda x: x[:3]/x[3] , posHomog))

pointCloud = removeHomog(posHomog)
print pointCloud

def writePointCloud(pointcloud):
	f = open(filename, "w")
	np.savetxt(f,pointcloud,fmt='%1.5e')
	f.close()

writePointCloud(pointCloud)
	



