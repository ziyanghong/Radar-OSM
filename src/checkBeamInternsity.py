import glob
import time
import cv2
import numpy as np
import os
import math
cwd = os.getcwd()
print cwd
from RadarUtils import compute_relative_pose, probabilistic_extract_points_from_scan,\
     oriented_surface_points, polar_to_cartesian,se2_to_SE2
from sklearn.cluster import DBSCAN, SpectralClustering
from rtree import index
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



CAR_REFLECTION_DIST = 3.0
MIN_CLOSE_OBJECT_THRESHOLD = 30
MAX_CLOSE_OBJECT_THRESHOLD = 50

#-----------------------------------------------------------------------------------------------------------------------
polar_data_dir = cwd +'/../data/radar/MulRan_Dataset/Riverside01/radar_zfill_six/'
car_data_dir = cwd +'/../data/radar/MulRan_Dataset/Riverside01/1601_200_radar_cart/'
odometry_file = cwd + '/../data/odometry/radar/Riverside01.txt'

polar_image_names = sorted(glob.glob(polar_data_dir + '*.png'))
cart_image_names = sorted(glob.glob(car_data_dir + '*.png'))

numImages = len(polar_image_names)
counter_data_skip = 1
k_frame_pointcloud = []
poses_old = []
accumulated_pointclouds = []

# thresholds = [10,20,30,40,50,60,70]
# saturation_beams = [46, 47, 48, 51,52,53, 78, 79, 80, 83, 84, 93, 95, 324, 325,326,327, 328, 332]
# interesting_beams = [166, 299, 300, 362, 363, 364]
thresholds = [50]
saturation_beams =[]
interesting_beams = []

start_frame = 339
max_range = 200.0
k_frame = 5
b_beam_checking = False
#-----------------------------------------------------------------------------------------------------------------------

def load_radar_odometry(path):
    tmpt = time.time()
    print 'Loading radar odometry...',

    odom_file = open(path,'r')
    lines = odom_file.readlines()
    odometry_list = []
    for line in lines:
        odom = np.vstack(  (np.asarray(line.split(),dtype=np.float32).reshape(3,4), np.array([0,0,0,1])))
        odometry_list.append(odom)
    print 'done in {0}s.'.format(time.time() - tmpt)
    print 'Read {0} odometry observations.'.format(len(odometry_list))
    return odometry_list

for thres in thresholds:
    odometry_list = load_radar_odometry(odometry_file)
    counter_data_skip = 1

    for i in range(start_frame, start_frame+k_frame):
        pose_current = odometry_list[i - 1]

        polar_img_path = polar_image_names[i]
        polar_img = cv2.imread(polar_img_path, 0)

        # # Cartesian image
        cart_img_path = cart_image_names[i]
        cart_img = cv2.imread(cart_img_path, 0)
        cv2.imwrite(cwd + '/scan2pointCloud/' + str(i + 1).zfill(4) + '_original_cart_img.png', cart_img)

        num_rows, num_cols = polar_img.shape
        num_beams = num_rows
        radar_resolution = 200.0 / num_cols
        remove_pixels = int(CAR_REFLECTION_DIST/radar_resolution)


        fig, ax = plt.subplots(figsize=(15,15))


        # To detect far range road boundary
        far_x = []
        far_y = []
        strong_beams_idx = []
        strong_beams_points_x = []
        strong_beams_points_y = []

        beam_means = np.mean(polar_img ,axis=1)
        beams_std = np.std(beam_means)
        beams_mean = np.mean(beam_means)
        relative_pose = compute_relative_pose(odometry_list[i - 2], pose_current)
        delta_x = relative_pose[2,3]
        delta_y = -relative_pose[0,3]
        delta_theta = -np.arctan2( relative_pose[0,2], relative_pose[0,0])
        velocity = np.array([delta_x / 0.25, delta_y / 0.25, delta_theta / 0.25])

        for j in range(polar_img.shape[0]):
            delta_T = 0.25 * ((j+1 - polar_img.shape[0]/2) /  polar_img.shape[0])
            se2_T0t = velocity*delta_T
            T0t = se2_to_SE2(se2_T0t)
            scan = polar_img[j,:].copy()
            beam_mean = np.mean(scan)
            # scan[0:366] = 0.0
            # if beam_mean >= (whole_image_mean + whole_image_std / 2):
            # if beam_mean >= (beams_mean + beams_std):
            peak_dist = np.sort(abs(scan[0:-1] - scan[1:]))
            b_strong_beam = False
            if beam_mean >= (beams_mean + beams_std) :
                b_strong_beam = True

                if b_beam_checking:
                    fig_beam, ax_beam = plt.subplots(figsize=(15, 15))
                    ax_beam.scatter(np.arange(0, scan.shape[0], 1), scan)
                    ax_beam.set_ylim((0,150))
                    fig_beam.savefig(cwd + '/scan2pointCloud/image_' + str(i+1).zfill(4) + '_beam_old_' + str(j+1)  + '.png')
                    strong_beams_idx.append(j)

                super_threshold_indices= scan < thres
                scan[super_threshold_indices] = 0
                peaks_idx = np.nonzero(scan)[0]
                peaks_idx = peaks_idx[0::8] # downsample here, otherwise we will have too many points
            else:
                peaks_idx = probabilistic_extract_points_from_scan(scan,max_range)
                # print('Number of points per beam:' + str(len(peaks_idx)))

            peak_per_beam_x = []
            peak_per_beam_y = []

            for p in peaks_idx:
                if scan[p] < thres:
                    continue
                azimuth = j
                distance = p
                x,y = polar_to_cartesian(azimuth, distance, radar_resolution, num_beams, T0t)
                peak_per_beam_x.append(x)
                peak_per_beam_y.append(y)

                far_x.append(x)
                far_y.append(y)
            if b_beam_checking:
                if b_strong_beam:
                    # colors = (j+1)*np.ones((len(peak_per_beam_y),3)) / 400.0
                    strong_beams_points_x.append(peak_per_beam_x)
                    strong_beams_points_y.append(peak_per_beam_y)
                    ax.scatter(peak_per_beam_x, peak_per_beam_y, s=8,  c= 'r')

        ax.scatter(far_x, far_y, s=8, c='k', alpha=0.2)

        # To detect close range road boundary
        close_x = []
        close_y = []
        for j in range(polar_img.shape[0]):
            delta_T = 0.25 * ((j+1 - polar_img.shape[0]/2) /  polar_img.shape[0])
            se2_T0t = velocity*delta_T
            T0t = se2_to_SE2(se2_T0t)
            scan = polar_img[j,0:366]
            scan[0:remove_pixels] = 0.0
            peaks_idx = probabilistic_extract_points_from_scan(scan,max_range)
            for p in peaks_idx:
                if scan[p] <= MIN_CLOSE_OBJECT_THRESHOLD or scan[p] >= MAX_CLOSE_OBJECT_THRESHOLD:
                    continue
                azimuth = j
                distance = p
                x,y = polar_to_cartesian(azimuth, distance, radar_resolution, num_beams, T0t)
                close_x.append(x)
                close_y.append(y)
        ax.scatter(close_x, close_y, s=8, c='r')
        ax.scatter(0, 0, s=10, c='k')
        print('Number of points:' + str(len(close_x) + len(far_x)))
        fig.savefig(cwd + '/scan2pointCloud/polar_raw_points' + str(thres) + '_' + str(i+1).zfill(4) + '.png')
        # cv2.imwrite(cwd + '/scan2pointCloud/polar_threshold_' + str(thres) + '_img.png', new_polar_img)

        # To align with the visual odometry coordinate
        all_x = far_x + close_x
        all_y = far_y + close_y
        pointcloud_np = np.zeros((4, len(all_x)))
        pointcloud_np[0,:] = -np.array(all_y)
        pointcloud_np[2,:] =  np.array(all_x)
        pointcloud_np[3,:] = 1

        k_frame_pointcloud.append(pointcloud_np)

        for counter, v in enumerate(strong_beams_idx):
            fig_beam_points, ax_beam_points = plt.subplots(figsize=(15, 15))
            ax_beam_points.scatter(all_x, all_y, s=8, c='b')
            ax_beam_points.scatter(strong_beams_points_x[counter], strong_beams_points_y[counter], s=8, c='r')

            fig_beam_points.savefig(cwd + '/scan2pointCloud/image_' + str(i+1).zfill(4) + '_highlight_beam_' + str(v+1) + '_''.png')

        if (counter_data_skip % k_frame) == 0:
            transformed_pointcloud = k_frame_pointcloud[-1]
            for j in range(k_frame-1):
                pose_old = poses_old[j]
                transformation_to_current_frame = compute_relative_pose(pose_old, pose_current)
                old_pointcloud = k_frame_pointcloud[j]
                temp_pointcloud = transformation_to_current_frame.dot(old_pointcloud)
                transformed_pointcloud = np.hstack((transformed_pointcloud, temp_pointcloud))

            # Convert back to our IROS/IJRR coordinate
            transformed_pointcloud_xy_coordinate = np.ones((4, transformed_pointcloud.shape[1]))
            transformed_pointcloud_xy_coordinate[0, :] = transformed_pointcloud[2, :]
            transformed_pointcloud_xy_coordinate[1, :] = -transformed_pointcloud[0, :]

            # Compute oriented surface points now......
            keypoints, normals = oriented_surface_points(transformed_pointcloud_xy_coordinate)

            # Draw image
            fig2, ax2 = plt.subplots()
            ax2.scatter(transformed_pointcloud_xy_coordinate[0,:], transformed_pointcloud_xy_coordinate[1,:],\
                        c='k',s=1, edgecolors='None')
            qv = ax2.quiver(keypoints[:,0], keypoints[:,1], normals[:,0], normals[:,1],normals[:,2])
            plt.colorbar(qv)
            ax2.set_xticks(np.arange(-200, 200, 40))
            ax2.set_yticks(np.arange(-200, 200, 20))

            fig_name = cwd + '/scan2pointCloud/kFrames_' + str(i+1).zfill(4) + '_' + '_pointcloud.png'
            fig2.savefig(fig_name,dpi=400)
            plt.clf()
            # displayFigSz = (7, 5)
            # fig_hist = plt.figure(figsize=displayFigSz)
            # ax_hist = fig_hist.add_subplot(111)
            # ax_hist.hist2d(transformed_pointcloud_xy_coordinate[0, :], transformed_pointcloud_xy_coordinate[1, :], bins=100)
            # directory = cwd + '/scan2pointCloud/' + str(i + 1).zfill(4) + '_'
            # fig_name = directory + '_hist.png'
            # fig_hist.savefig(fig_name, dpi=400)
            # reset these variables
            k_frame_pointcloud = []
            poses_old = []
        else:
            poses_old.append(pose_current)
        counter_data_skip += 1


if __name__ == '__main__':
    print'check beam intensity'