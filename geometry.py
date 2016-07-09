import numpy as np

#origin is camera position. z is direction camera is looking. x is to the right. y is up.
#Waiiiiiit. That's a left handed cooridnate system? Huh. Whatever. May come out mirrore
PCameraHomog = np.array([0.,0.,0.,1.])
#Baseline distance of 
#Let's use units of meters
PLaser = np.array([ 0.3 ,0,0])

#I have measured my angle from -x going clockwise. God that is dumb.
laserAngle = 60.
laserRadian = laserAngle *np.pi/180.

PLaserHomog = np.append(PLaser, [1.])
upDirHomog = np.array([0.,0.,1.,0.])
laserDirHomog = np.array([-1.* np.cos(laserRadian), np.sin(laserRadian), 0., 0.])

planeMat = np.stack((PLaserHomog, upDirHomog, laserDirHomog))

def colminor(mat,j):
	subMat = np.delete(mat, j, axis=1)
	return (-1.)**j * np.linalg.det(subMat)

#The homogenous vector describing the plane coming off of the line laser. p dot x = 0 if x is on plane
laserPlaneHomog = np.array(map(lambda j: colminor(planeMat, j) , range(4)))

#Should all be zero
print np.dot(laserPlaneHomog, laserDirHomog)
print np.dot(laserPlaneHomog, upDirHomog)
print np.dot(laserPlaneHomog, PLaserHomog)

def pixelDir(x,y):
	# pix / f = objsize / objdist
	# f = pix * objdist / objsize
	f = 100. #camera Width of 1m object at 1m in pixels, or 8m object at 8m. 
	return np.array([ x / f ,  y / f , 1., 0.])

cameraRay = pixelDir(10,20)

#pos is on line between camera pos and ray and lies on laserplane. Hence pos dot plane = 0, which you can see will happen
posHomog = np.dot(cameraRay, laserPlaneHomog) * PCameraHomog - np.dot(PCameraHomog, laserPlaneHomog) * cameraRay
print posHomog

def removeHomog(x):
	return x[:3]/x[3]

pos3 = removeHomog(posHomog)

print pos3


