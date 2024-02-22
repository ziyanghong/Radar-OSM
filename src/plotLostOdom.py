import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


odom_file = '/media/data/RadarOSM/data/odometry/libviso2_stereo/00.txt'
# odom_file = '/home/hong/Documents/Lost/data/odometry/radar/2019-01-16-13-09-37-radar-oxford-10k.txt'
gt_file = '/media/data/RadarOSM/data/map_trajectories/00.txt'

vecX = []
vecY = []
vecZ = []
# read radar timestamp
file = open(gt_file, "r")
lines = file.readlines()
for line in lines:
	line  = line.split(' ')
	vecX.append(float(line[3]))
	vecY.append(float(line[7]))
	vecZ.append(float(line[11]))

vecX = np.asarray(vecX)
vecY = np.asarray(vecY)
vecZ = np.asarray(vecZ)

# print vecX
fig = plt.figure()
  
# syntax for 3-D projection
ax = plt.axes(projection ='3d')

# plotting
ax.plot3D(vecX, vecY, vecZ, 'green')
ax.set_title('3D line plot geeks for geeks')
# Set axes label
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()