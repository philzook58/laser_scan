
import numpy as np

PLaser = np.array([0.35,0,0])
focal = 860. 
PCameraHomog = np.array([0.,0.,0.,1.])


points = np.array([[1,0,1],
	[0,0,10],
	[90.35,90,90],
	[.35,0,1]])

#points = np.random.randn(6,3)

print "Points"
print points
def pixels(pointcloud):
	# pix / f = objsize / objdist
	# f = pix * objdist / objsize
	f = focal #100. #camera Width of 1m object at 1m in pixels, or 8m object at 8m. 
	return focal*np.array([pointcloud[:,0]/pointcloud[:,2],pointcloud[:,1]/pointcloud[:,2]]).T 

angles = 180/np.pi* np.asarray(map( lambda point: np.pi - np.arctan2( (point-PLaser)[2] , (point-PLaser)[0]  ), points))
pixel = pixels(points)
pointcloud = np.array([pixel[:,0], pixel[:,1], angles ]).T 


print "Pointcloud pixel pixel angle"
print pointcloud



laserRadian = pointcloud[:,2] * np.pi / 180


PLaserHomog = np.append(PLaser, [1.])
upDirHomog = np.array([0.,1.,0.,0.])
#laserDirHomog = np.array([-1.* np.cos(laserRadian), np.sin(laserRadian), 0., 0.])
zeros = np.zeros_like(laserRadian)
ones = np.ones_like(laserRadian)
laserDirHomog = np.array([-1.*np.cos(laserRadian), zeros, np.sin(laserRadian),zeros]).T

print "laser direction"
print laserDirHomog
def planeMat(laserDirHomog):
	#print np.stack((PLaserHomog, upDirHomog, laserDirHomog))
 	return np.stack((PLaserHomog, upDirHomog, laserDirHomog))

def colminor(mat,j):
	subMat = np.delete(mat, j, axis=1)
	#print subMat
	return (-1.)**j * np.linalg.det(subMat)


laserPlaneHomogs = map(lambda laserDirH: np.asarray(map(lambda j: colminor(planeMat(laserDirH), j) , range(4))) , laserDirHomog)
laserPlaneHomogs = np.asarray(laserPlaneHomogs)

print"laserplanes"
print laserPlaneHomogs




def pixelDir(pointcloud):
	# pix / f = objsize / objdist
	# f = pix * objdist / objsize
	f = focal #100. #camera Width of 1m object at 1m in pixels, or 8m object at 8m. 
	return np.array([pointcloud[:,0]/f,pointcloud[:,1]/f,ones,zeros]).T #I could have replaced the ones with f.

pixelDirs = pixelDir(pointcloud)
print "camera rays"
print pixelDirs


#posHomog = map(lambda laserPlaneHomog: map(lambda cameraRay: np.dot(cameraRay, laserPlaneHomog) * PCameraHomog - np.dot(PCameraHomog, laserPlaneHomog) * cameraRay, pixelDirs), laserPlaneHomogs)
posHomog = np.zeros_like(laserPlaneHomogs)
'''
print pointcloud[-1,:]
print pixelDirs[-1,:]
print laserPlaneHomogs[-1,:]
print PCameraHomog
print np.dot(pixelDirs[-1,:], laserPlaneHomogs[-1,:]) * PCameraHomog - np.dot(PCameraHomog, laserPlaneHomogs[-1,:]) * pixelDirs[-1,:]
'''
for i in range(pixelDirs.shape[0]):
	posHomog[i,:] = np.dot(pixelDirs[i,:], laserPlaneHomogs[i,:]) * PCameraHomog - np.dot(PCameraHomog, laserPlaneHomogs[i,:]) * pixelDirs[i,:]
print "homogenopos"
print posHomog
#posHomog = np.asarray(posHomog)

def removeHomog(posHomog):
	return np.asarray( map(lambda x: x[:3]/x[3] , posHomog))

pointCloud = removeHomog(posHomog)
print "reconstructed points"
print pointCloud
print "original points"
print points