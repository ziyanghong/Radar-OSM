import os 
import glob
import csv
import math
import numpy as np



""" MulRan """
# project_path= '/mnt/gpuServerFolder/diskB/'
# data_set_name = 'MulRan_Dataset/'
# sequence_name = 'KAIST01'
# sequence_name = 'KAIST02'
# sequence_name = 'KAIST03'

# sequence_name = 'DCC01'
# sequence_name = 'DCC02'
# sequence_name = 'DCC03'

# sequence_name = 'Riverside01'
# sequence_name = 'Riverside02'
# sequence_name = 'Riverside03'

# sequence_name = 'Sejong01'
# sequence_name = 'Sejong02'

""" Oxford """
project_path= '/media/hong/DiskC/'
data_set_name = 'Oxford_Radar_RobotCar_Dataset/'
# sequence_name = '2019-01-10-11-46-21-radar-oxford-10k'
# sequence_name = '2019-01-10-12-32-52-radar-oxford-10k'
# sequence_name = '2019-01-16-11-53-11-radar-oxford-10k'
# sequence_name = '2019-01-16-13-09-37-radar-oxford-10k'
# sequence_name = '2019-01-17-13-26-39-radar-oxford-10k'
# sequence_name = '2019-01-18-14-14-42-radar-oxford-10k'
# sequence_name = '2019-01-18-15-20-12-radar-oxford-10k'
# sequence_name = '2019-01-18-14-46-59-radar-oxford-10k'

data_sequence_path = project_path + data_set_name + sequence_name 
odom_path =data_sequence_path + '/ro/odometry/1/our_result_odometry.csv'
gps_path = data_sequence_path + '/gps/gps.csv'
radar_timestamp_file = data_sequence_path + '/radar.timestamps'

output_name = sequence_name + '.txt'
output_gps_file = '/home/hong/Documents/RadarOSM/data/map_trajectories/' + output_name
output_odom_file = '/home/hong/Documents/RadarOSM/data/odometry/radar/' + output_name

start_frame = 0
end_frame = 1000000

lat_lon_alt = []
gps_timestamps = []
radar_timestamps = []

# read gps data
with open(gps_path) as csvfile:
	csvReader = csv.reader(csvfile, delimiter=',')
	if 'Oxford' in data_set_name:
		next(csvReader)
	for row in csvReader:
		timestamp_str = row[0]
		# print int(timestamp_str)
		gps_timestamps.append(int(timestamp_str))
		if 'Oxford' in data_set_name:
			lat_lon_alt.append([float(row[2]), float(row[3]), float(row[4])])

		elif 'MulRan' in data_set_name:
			lat_lon_alt.append([float(row[1]), float(row[2]), float(row[3])])
gps_timestamps = np.asarray(gps_timestamps) 

# read radar timestamp
file = open(radar_timestamp_file, "r")
lines = file.readlines()
for line in lines:
	line  = line.split(' ')
	# print line[0]
	radar_timestamps.append(int(line[0]))

# sync gps and radar
sync_gps_file = open(output_gps_file, "w")
for count, t in enumerate(radar_timestamps):
	# print t
	if count >= start_frame and count <= end_frame:
		idx_gps = (np.abs(gps_timestamps - t)).argmin()
		print (lat_lon_alt[idx_gps])
		line = str(lat_lon_alt[idx_gps][0]) + ' ' + str(lat_lon_alt[idx_gps][1]) + ' ' + str(lat_lon_alt[idx_gps][2])
		for i in range(27):
			line  = line + ' 0'
		line = line + '\n'	
		sync_gps_file.write(line)
		# sync_gps.append(lat_lon_alt[idx_gps])
sync_gps_file.close()
file.close()

# convert our odometry to libviso2 format
odom_file = open(odom_path, "r")
formatted_odom_file = open(output_odom_file, "w")

# # First pose
# pose = np.eye(4)
# line = ' '.join(map(str, pose[0:3,0:4].flatten()))
# line = line + '\n'	
# formatted_odom_file.write(line)

with open(odom_path) as csvOdom:
	csvReader = csv.reader(csvOdom, delimiter=',')
	next(csvReader)
	for count, row in enumerate(csvReader):
		if count >= start_frame and count <= end_frame:
			x = float(row[1])
			y = float(row[2])
			yaw = float(row[3])
			# T_a_c = np.array([[math.cos(yaw), math.sin(yaw), 0, x],
			# 	              [-math.sin(yaw), math.cos(yaw), 0, y],
			#     	          [             0,             0, 1, 0],
			#         	      [             0,             0, 0, 1]])
			# print T_a_c
			# T_b_c = np.dot(T_a_c, T_b_a)
			# print T_b_c

			pose = np.array([[ math.cos(-yaw), 0, math.sin(-yaw), -y],
				             [0,               1,              0,  0],
			    	         [-math.sin(-yaw), 0, math.cos(-yaw),  x],
			        	      [             0, 0,              0,  1]])

			line = ' '.join(map(str, pose[0:3,0:4].flatten()))
			line = line + '\n'	

			formatted_odom_file.write(line)
# print T_b_a



