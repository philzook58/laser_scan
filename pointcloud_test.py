#import pcl
import numpy as np


#p = pcl.PointCloud(np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32))
#p.to_file("cloudtest.pcd")


f = open("cloudtest.pcd", "w")

f.write('''# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 213
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 213
DATA ascii
''')

p = np.array([[1, 2000000000000, .03], [300.656, 4, 5]], dtype=np.float32)
#p.tofile(f, sep=" ")
#np.array_str(p , )
np.savetxt(f,p,fmt='%1.5e')
f.close()