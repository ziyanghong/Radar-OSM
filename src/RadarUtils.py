import glob
from scipy.signal import find_peaks
import cv2
import numpy as np
import sys
import time
import utm
from numba import jit, njit
from rtree import index
from ssc import ssc
import multiprocessing as mp


from matplotlib.collections import LineCollection
from MapBuilder import mercatorProj
from TransformUtils import se2_to_SE2
from PIL import Image
from matplotlib.ticker import PercentFormatter

import os

cwd = os.getcwd()
GRID_STEP_SIZE = 0.5
CAR_REFLECTION_DIST = 3.0
MIN_CLOSE_OBJECT_THRESHOLD = 30
MAX_CLOSE_OBJECT_THRESHOLD = 50
RADAR_INTENSITY_THRESH = 50
debugFlag = True

def build_rtree_points(points):
    points_rtree = index.Index()
    # tmpt = time.time()
    for i in range(points.shape[1]):
        p = points[:,i]
        points_rtree.insert(i, (p[0], p[1], p[0], p[1]))
    # print '\nBuild points rtree in {0}s.'.format(time.time() - tmpt)

    return points_rtree

def compute_relative_pose(w_t_a, w_t_b):
    # b_t_a = (w_t_b)^-1 * (w_t_a)

    b_t_w = np.eye(4)
    w_translation_b = w_t_b[0:3,3].reshape(3,1)
    b_rotation_w = w_t_b[0:3,0:3].transpose()
    b_translation_w = -b_rotation_w.dot(w_translation_b)

    b_t_w[0:3,0:3] = b_rotation_w
    b_t_w[0:3,3] = b_translation_w.reshape(3)
    b_t_a = b_t_w.dot(w_t_a)
    return b_t_a



def load_gps_data(options):
    gps_file = open(options.gps_data,'r')
    lines = gps_file.readlines()
    gps_list = []
    for line in lines:
        line = line.split()
        latlon = [float(line[0]), float(line[1])]
        gps_list.append(latlon)
    return gps_list

def load_radar_odometry(options):
    tmpt = time.time()
    print 'Loading radar odometry...',

    odom_file = open(options.odometry,'r')
    lines = odom_file.readlines()
    odometry_list = []
    for line in lines:
        odom = np.vstack(  (np.asarray(line.split(),dtype=np.float32).reshape(3,4), np.array([0,0,0,1])))
        odometry_list.append(odom)
    print 'done in {0}s.'.format(time.time() - tmpt)
    print 'Read {0} odometry observations.'.format(len(odometry_list))
    return odometry_list

@njit
def extract_pointcloud_old(img, max_range, selected_max_distance, radar_resolution):

    num_rows, num_cols = img.shape
    topK = 20
    pointcloud = np.zeros((4, topK*num_rows))
    max_pixel = int(selected_max_distance / max_range * num_cols)


    remove_pixels = int(CAR_REFLECTION_DIST/radar_resolution)
    counter = 0
    for i in range(num_rows):
        scan = img[i,0:max_pixel]
        scan[0:remove_pixels] = 0.0 # remove car self-reflection
        # points_idx = probabilistic_extract_points_from_scan(scan,max_range)
        # points_idx = topK_intensity_extract_points_from_scan(scan, topK)
        points_idx = scan.argsort()[-(topK + 1):-1]

        for j in range(len(points_idx)):
            # if scan[points_idx[j]] < 60:
            #     continue
            theta = -i * 2 * np.pi / num_rows
            x = points_idx[j]*radar_resolution*np.cos(theta) # in our IROS/IJRR coordinate
            y = points_idx[j]*radar_resolution*np.sin(theta) # in our IROS/IJRR coordinate

            # To align with the visual odometry coordinate
            vision_coordiante_x = -y
            vision_coordiante_y = 0
            vision_coordiante_z = x
            pointcloud[:,counter] = np.array([vision_coordiante_x,vision_coordiante_y,vision_coordiante_z,1])
            counter += 1
            # pointcloud.append([vision_coordiante_x,vision_coordiante_y,vision_coordiante_z,1])

    # pointcloud = np.asarray(pointcloud).transpose()
    # pointcloud = pointcloud[:,1:counter]
    return pointcloud

def single_extract(polar_img, max_range, selected_max_distance, \
                       radar_resolution, pose_current, pose_previous):
    num_rows, num_cols = polar_img.shape
    num_beams = num_rows
    remove_pixels = int(CAR_REFLECTION_DIST / radar_resolution)

    selected_max_idx = int(float(selected_max_distance) / float(radar_resolution))
    # fig, ax = plt.subplots(figsize=(15, 15))

    # To detect far range road boundary
    far_x = []
    far_y = []


    relative_pose = compute_relative_pose(pose_current, pose_previous)
    delta_x = relative_pose[2, 3]
    delta_y = -relative_pose[0, 3]
    delta_theta = -np.arctan2(relative_pose[0, 2], relative_pose[0, 0])
    velocity = np.array([delta_x / 0.25, delta_y / 0.25, delta_theta / 0.25])
    for j in range(polar_img.shape[0]):
        delta_T = 0.25 * ((j + 1 - polar_img.shape[0] / 2 ) / polar_img.shape[0])
        se2_T0t = velocity * delta_T
        T0t = se2_to_SE2(se2_T0t)
        scan = polar_img[j, 0:selected_max_idx].copy()
        scan[0:remove_pixels] = 0.0 # remove car self-reflection

        # peaks_idx = topK_intensity_extract_points_from_scan(scan, topK)
        peaks_idx = probabilistic_extract_points_from_scan(scan, radar_resolution)

        # beam_mean = np.mean(scan)
        # if beam_mean >= (beams_mean + beams_std):
        #     b_strong_beam = True
        #     if b_beam_checking:
        #     #     fig_beam, ax_beam = plt.subplots(figsize=(15, 15))
        #     #     ax_beam.scatter(np.arange(0, scan.shape[0], 1), scan)
        #     #     ax_beam.set_ylim((0, 150))
        #     #     fig_beam.savefig(
        #     #         cwd + '/scan2pointCloud/image_' + str(i + 1).zfill(4) + '_beam_old_' + str(j + 1) + '.png')
        #     #     strong_beams_idx.append(j)
        #
        #     super_threshold_indices = scan < RADAR_INTENSITY_THRESH
        #     scan[super_threshold_indices] = 0
        #     peaks_idx = np.nonzero(scan)[0]
        #     peaks_idx = peaks_idx[0::8]  # downsample here, otherwise we will have too many points
        # else:
        #     peaks_idx = probabilistic_extract_points_from_scan(scan, radar_resolution)
        # peaks_idx = scan.argsort()[-(topK + 1):-1]


        # if i > 60:
        #     print 'debug'
        peak_per_beam_x = []
        peak_per_beam_y = []

        for p in peaks_idx:
            if scan[p] < RADAR_INTENSITY_THRESH:
                continue
            azimuth = j
            distance = p
            x, y = polar_to_cartesian(azimuth, distance, radar_resolution, num_beams, T0t)
            peak_per_beam_x.append(x)
            peak_per_beam_y.append(y)

            far_x.append(x)
            far_y.append(y)
    return far_x,far_y

# def multi_extract(delta_theta, velocity, scan):

def extract_pointcloud(i, polar_img, max_range, selected_max_distance, \
                       radar_resolution, pose_current, pose_previous):
    all_x_y = []


    topK = 60
    num_rows, num_cols = polar_img.shape
    remove_pixels = int(CAR_REFLECTION_DIST / radar_resolution)

    selected_max_idx = int(float(selected_max_distance) / float(radar_resolution))


    relative_pose = compute_relative_pose(pose_current, pose_previous)
    delta_x = relative_pose[2, 3]
    delta_y = -relative_pose[0, 3]
    delta_theta = -np.arctan2(relative_pose[0, 2], relative_pose[0, 0])
    velocity = np.array([delta_x / 0.25, delta_y / 0.25, delta_theta / 0.25])
    for j in range(polar_img.shape[0]):
        delta_T = 0.25 * ((j + 1 - polar_img.shape[0] / 2 ) / polar_img.shape[0])
        se2_T0t = velocity * delta_T
        T0t = se2_to_SE2(se2_T0t)
        scan = polar_img[j, 0:selected_max_idx].copy()
        scan[0:remove_pixels] = 0.0 # remove car self-reflection
        # peaks_idx = topK_intensity_extract_points_from_scan(scan, topK)
        peaks_idx = probabilistic_extract_points_from_scan(scan, radar_resolution)

        for p in peaks_idx:
            if scan[p] < RADAR_INTENSITY_THRESH:
                continue
            azimuth = j
            distance = p
            x, y = polar_to_cartesian(azimuth, distance, radar_resolution, num_rows, T0t)
            all_x_y.append([x,y])

    x_y_np = np.array(all_x_y)
    pointcloud_np = np.zeros((4, len(all_x_y)))
    pointcloud_np[0, :] = -np.array(x_y_np[:,1])
    pointcloud_np[2, :] = np.array(x_y_np[:,0])
    pointcloud_np[3, :] = 1
    return pointcloud_np


    

def polar_to_cartesian(azimuth, range, radar_resolution, num_beams, T0t=None):
    if T0t is None:
        T0t= np.eye(3)
    theta = -azimuth * 2 * np.pi / num_beams
    x = range * radar_resolution * np.cos(theta)  # in our IROS/IJRR coordinate
    y = range * radar_resolution * np.sin(theta)  # in our IROS/IJRR coordinate
    corrected_measurement = T0t.dot(np.array([[x], [y], [1]]))
    x = corrected_measurement[0, 0]
    y = corrected_measurement[1, 0]
    return x,y

def divide_grids(points, grid_resolution):
    grids = []
    return grids

def points_nms(points, normals):
    num_ret_points = 150
    tolerance = 0.2

    nms_points = []
    nms_normals = []
    keypoints = []
    x_min = np.min(points[0,:]) - 10.0
    y_min = np.min(points[1,:]) - 10.0
    x_max = np.max(points[0,:]) + 10.0
    y_max = np.max(points[1,:]) + 10.0
    cols = int(x_max + abs(x_min))
    rows = int(y_max + abs(y_min))
    x_shift = abs(x_min)
    y_shift = abs(y_min)
    for i in range(points.shape[1]):
        kpt = cv2.KeyPoint(x=(points[0,i] + x_shift), y=(points[1,i] + y_shift), _size=0.0, \
                           _response=normals[2,i], _octave=0, _class_id=0)
        keypoints.append(kpt)

    # keypoints should be sorted by strength in descending order
    # before feeding to SSC to work correctly
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)


    selected_keypoints, selected_idxs = ssc(
        keypoints, num_ret_points, tolerance, cols, rows
    )

    for idx in selected_idxs:
        nms_points.append(points[:,idx])
        nms_normals.append(normals[:,idx])
    nms_points = np.asarray(nms_points).transpose()
    nms_normals = np.asarray(nms_normals).transpose()
    return nms_points, nms_normals

def oriented_surface_points(pointcloud):
    """
    Input:
    @pointcloud: 4xM matrix
    Output:
    @keypoints: 3xN matrix with N << M [x,y,1]^T
    @normals: 3xN matrix with the eigenvectors that corresponds [x,y,dominance]^T
     to the smallest eigenvalue
    """



    keypoints = []
    normals = []
    continuity_factor = 5.0
    close_threshold = 30
    car_reflection_distance = 5.0
    points_rtree = build_rtree_points(pointcloud)

    tmpt = time.time()

    x_min = np.min(pointcloud[0,:])
    x_max = np.max(pointcloud[0,:])
    y_min = np.min(pointcloud[1,:])
    y_max = np.max(pointcloud[1,:])
    # t = np.arange(1., 10, 1)**2
    t_far = np.arange(close_threshold, 150, 10)
    # t_close =  np.arange(2.0, 4.47, 0.6)**2
    t_close =  np.arange(5.0, close_threshold, 5)

    t = np.hstack((t_close,t_far))

    x_centers = np.hstack((-t, t))
    y_centers = np.hstack((-t, t))
    scale_factor = 0.5

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    xx, yy = np.meshgrid(x_centers, y_centers)
    xx = xx.flatten()
    yy = yy.flatten()
    # fig, ax = plt.subplots()
    # ax.scatter(xx, yy, s=1)
    # fig_name = cwd + '/scan2pointCloud/grid.png'
    # fig.savefig(fig_name, dpi=400)

    for i in range(xx.shape[0]):
        x = xx[i]
        y = yy[i]
        if abs(x) < car_reflection_distance and abs(y) < car_reflection_distance:
            continue

        if x < close_threshold:
            x_width = 5.0
        else:
            x_width = 10.0
        if y < close_threshold:
            y_width = 5.0
        else:
            y_width = 10.0

        # x_width = 5.0
        # y_width = 5.0

        intersect_idx = list(points_rtree.intersection((x-x_width, y-y_width, \
                                                        x+x_width, y+y_width)))

        # Too few points, ignore this we are not confident about this
        if len(intersect_idx) < 5:
            continue
        intersect_points = pointcloud[:,intersect_idx]
        points_xy = intersect_points[0:2,:]

        # Check curve continuity
        sort_x = np.sort(points_xy[0,:])
        sort_y = np.sort(points_xy[1,:])
        dist_x = np.sort(abs(sort_x[0:-1] - sort_x[1:]))
        dist_y = np.sort(abs(sort_y[0:-1] - sort_y[1:]))
        max_dist_x = dist_x[-1]
        max_dist_y = dist_y[-1]
        inliers_mean_x = np.mean(dist_x[0:(dist_x.shape[0]-2)])
        inliers_mean_y = np.mean(dist_y[0:(dist_x.shape[0]-2)])

        # Compute local statistics
        covariance = np.cov(points_xy)
        mean = np.mean(points_xy,axis=1)
        w,v = np.linalg.eig(covariance)
        min_eig_idx = np.argmin(w)
        max_eig_idx = np.argmax(w)
        eigen_value_dominance = np.array([w[max_eig_idx] / w[min_eig_idx]])
        normal_vector = v[:,min_eig_idx]


        # fig_grid, ax_grid = plt.subplots()
        # ax_grid.scatter(pointcloud[0,:], pointcloud[1,:], s=0.5, c='b')
        # ax_grid.scatter(points_xy[0,:], points_xy[1,:], s=0.5, c='r')
        # ax_grid.quiver(mean[0], mean[1], normal_vector[0], normal_vector[1])
        # ax_grid.scatter(x, y, s=1, c='k')
        # disBox = []
        # disBox.append([[x-x_width, y-y_width], [x-x_width, y+y_width]])
        # disBox.append([[x-x_width, y+y_width], [x+x_width, y+y_width]])
        # disBox.append([[x+x_width, y+y_width], [x+x_width, y-y_width]])
        # disBox.append([[x+x_width, y-y_width], [x-x_width, y-y_width]])
        # ax_grid.add_collection(LineCollection(disBox, transOffset=ax.transData,
        #                                   linewidths=0.2, colors='purple'))
        # fig_name = cwd + '/scan2pointCloud/grid_'+ str(i).zfill(4) + '.png'
        # title_str = '\nnumber of points: ' +  str(len(intersect_idx)) + \
        #             '\ninliers_mean_x: ' + '{0:3.2f}'.format(inliers_mean_x) + \
        #             ' inliers_mean_y: ' + '{0:3.2f}'.format(inliers_mean_y) + \
        #             ' factor:' '{0:3.2f}'.format(continuity_factor)+ \
        #             '\nmax_dist_x:' + '{0:3.2f}'.format(max_dist_x) + \
        #             ' max_dist_y:' + '{0:3.2f}'.format(max_dist_y) + \
        #             ' reject code: '

        # # A lot of points
        # if len(intersect_idx) >= 50:
        #     # Insert new point
        #     keypoints.append(mean)
        #     normals.append(np.hstack((normal_vector, len(intersect_idx))))
        #     plt.title(title_str)
        #     fig_grid.savefig(fig_name, dpi=400)
        #     continue

        # if max_dist_x > inliers_mean_x*max([continuity_factor,abs(mean[0])]):
        #     plt.title(title_str + str(1) + 'x: ' + str(inliers_mean_x*max([continuity_factor,abs(mean[0])])))
        #     fig_grid.savefig(fig_name, dpi=400)
        #     continue
        # elif max_dist_y > inliers_mean_y*max([continuity_factor,abs(mean[1])]):
        #     plt.title(title_str + str(1) + 'y: ' + str(inliers_mean_y*max([continuity_factor,abs(mean[1])])))
        #     fig_grid.savefig(fig_name, dpi=400)
        #     continue

        # Check if most of the points are concentrated, due to drift they might be the same point,
        # and we should not take this point
        if (inliers_mean_x < 0.2) and (inliers_mean_y <  0.2) and len(intersect_idx) < 10:
            # plt.title(title_str + str(2))
            # fig_grid.savefig(fig_name, dpi=400)
            continue

        # Both direction do not reveal any geometrical information
        if covariance[0,0] < 3.0 and covariance[1,1] < 3.0:
            # plt.title(title_str + str(3))
            # fig_grid.savefig(fig_name, dpi=400)
            continue

        # # Both direction spread too random, it is not a curve shape
        # if covariance[0,0] > 5.0 and covariance[1,1] > 5.0:
        #     plt.title(title_str + str(4))
        #     fig_grid.savefig(fig_name, dpi=400)
        #     continue

        # # Check is one direction is dominant, if not,
        # # it is just a cluster of points without geometrical information
        # if eigen_value_dominance < 3:
        #     plt.title(title_str + str(5))
        #     fig_grid.savefig(fig_name, dpi=400)
        #     continue


        # Insert new point
        keypoints.append(mean)
        normals.append(np.hstack((normal_vector,len(intersect_idx) )))
        # plt.title(title_str)
        # fig_grid.savefig(fig_name, dpi=400)

        # fig_grid, ax_grid = plt.subplots()
        # ax_grid.scatter(pointcloud[0,:], pointcloud[1,:], s=0.5, c='b')
        # ax_grid.scatter(points_xy[0,:], points_xy[1,:], s=0.5, c='r')
        # ax_grid.quiver(mean[0], mean[1], normal_vector[0], normal_vector[1])
        # ax_grid.scatter(x, y, s=1, c='k')
        # disBox = []
        # disBox.append([[x-x_width, y-y_width], [x-x_width, y+y_width]])
        # disBox.append([[x-x_width, y+y_width], [x+x_width, y+y_width]])
        # disBox.append([[x+x_width, y+y_width], [x+x_width, y-y_width]])
        # disBox.append([[x+x_width, y-y_width], [x-x_width, y-y_width]])
        # ax_grid.add_collection(LineCollection(disBox, transOffset=ax.transData,
        #                                   linewidths=0.2, colors='purple'))
        # fig_name = cwd + '/scan2pointCloud/grid_'+ str(i).zfill(4) + '.png'
        # title_str = ' \nxx: ' + '{0:3.2f}'.format(covariance[0,0]) + \
        #             ' xy: ' + '{0:3.2f}'.format(covariance[0,1]) + \
        #             ' yx: ' + '{0:3.2f}'.format(covariance[1,0]) + \
        #             ' yy: ' + '{0:3.2f}'.format(covariance[1,1]) + \
        #             '\ninliers_mean_x: ' '{0:3.2f}'.format(inliers_mean_x) + \
        #             '\ninliers_mean_y: ' '{0:3.2f}'.format(inliers_mean_y) + \
        #             ' max_dist_x:' + '{0:3.2f}'.format(max_dist_x) + \
        #             ' max_dist_y:' + '{0:3.2f}'.format(max_dist_y)
        #
        # plt.title(title_str)
        # fig_grid.savefig(fig_name, dpi=400)




    keypoints = np.array(keypoints).transpose()
    keypoints = np.vstack((keypoints,np.ones((1,keypoints.shape[1]))))
    normals = np.array(normals).transpose()
    # print 'Compute surface points in {0}s.'.format(time.time() - tmpt)
    # print 'Total oriented surface points: {0}'.format(keypoints.shape[1])

    return keypoints, normals
def probablistic_points_from_whole_image(image):
    peaks_mean = np.mean(image)
    peaks_std = np.std(image)
    return peaks_mean,peaks_std

@njit
def pick_top_peaks(scan, peaks_idx, max_point_per_beam):
    peaks_mean = np.mean(scan[peaks_idx])
    peaks_std = np.std(scan[peaks_idx])
    num_points_per_beam_counter = 0
    points_idx = []
    for j in range(len(peaks_idx)):
        pk = scan[peaks_idx[j]]
        if pk > peaks_mean + peaks_std:
            num_points_per_beam_counter += 1
            points_idx.append(peaks_idx[j])
        if num_points_per_beam_counter >= max_point_per_beam:
            break
    return points_idx
    
def probabilistic_extract_points_from_scan(scan, resolution):
    """
    input:
        resolution: meters/pixel
    """
    peak_prominence = 10
    max_point_per_beam = 30
    peak_distance = 4.0 # meters

    pixel_dist = peak_distance / resolution
    peaks_idx, _ = find_peaks(scan, prominence=peak_prominence, distance=pixel_dist)
    # peaks_mean = np.mean(scan[peaks_idx])
    # peaks_std = np.std(scan[peaks_idx])
    # num_points_per_beam_counter = 0

    # points_idx = []
    # for j in range(len(peaks_idx)):
    #     pk = scan[peaks_idx[j]]
    #     if pk > peaks_mean + peaks_std:
    #         num_points_per_beam_counter += 1
    #         points_idx.append(peaks_idx[j])
    #     if num_points_per_beam_counter >= max_point_per_beam:
    #         break
    points_idx = pick_top_peaks(scan, peaks_idx, max_point_per_beam)

    return points_idx

@njit
def topK_intensity_extract_points_from_scan(scan,topK):
    indices = scan.argsort()[-(topK+1):-1]
    return indices

def batch_utm(points_utm, gt_utm, projScale, ptCoordOrigin):
    num_points = points_utm.shape[1]

    mercator_visualization_x = []
    mercator_visualization_y = []
    EARTH_RAD_EQ = 6378.137 # in km

    for i in range(num_points):
        pt_utm = points_utm[:, i].transpose()
        pt_latlon = utm.to_latlon(pt_utm[0], pt_utm[1], gt_utm[2], gt_utm[3])
        # pt_mercator_visualization = mercatorProj(pt_latlon, projScale) - ptCoordOrigin
        pt_mercator_visualization = np.array([projScale * pt_latlon[1] * (np.pi / 180.0) * EARTH_RAD_EQ,\
                  projScale * EARTH_RAD_EQ * np.log(np.tan((90.0 + pt_latlon[0]) * (np.pi / 360.0)))]) - ptCoordOrigin

        mercator_visualization_x.append(pt_mercator_visualization[0])
        mercator_visualization_y.append(pt_mercator_visualization[1])
    return mercator_visualization_x,mercator_visualization_y

def visualize_pointcloud_on_map(frame_id, gps_list, projScale, ptCoordOrigin,\
                                keypoints, points_normal, raw_points, mapgraph):
    # -----------------------------------------Visualization method 1---------------------------------------------------
    import matplotlib
    from matplotlib.collections import PathCollection

    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    way_line_width = 0.0005

    print 'visualize pointcloud on_map.'
    print 'latlon of tInd = {0}:'.format(frame_id)

    gt_latlon_curr = gps_list[frame_id]
    print gt_latlon_curr


    gt_current = mercatorProj(gt_latlon_curr, projScale) - ptCoordOrigin
    gt_previous = mercatorProj(gps_list[frame_id - 1], projScale) - ptCoordOrigin
    x = gt_current[0] - gt_previous[0]
    y = gt_current[1] - gt_previous[1]
    heading_mercator = np.arctan2(y, x)
    transformation = np.asarray([[np.cos(heading_mercator), -np.sin(heading_mercator), 0], \
                                 [np.sin(heading_mercator), np.cos(heading_mercator), 0], \
                                 [0, 0, 1]])
    transformed_keypoints = np.dot(transformation, keypoints[0:3, :]) / 1000.0
    keypoints_projection = transformed_keypoints[0:2, :] + np.array([[gt_current[0]], [gt_current[1]]])
    transformed_rawpoints = np.dot(transformation, raw_points[0:3, :]) / 1000.0
    rawpoints_projection = transformed_rawpoints[0:2, :] + np.array([[gt_current[0]], [gt_current[1]]])
    normals_global = np.array([[np.cos(heading_mercator), -np.sin(heading_mercator)],
                               [np.sin(heading_mercator),  np.cos(heading_mercator)]]).dot(points_normal[0:2,:])

    fig_crop = plt.figure(dpi = 200,figsize=(5,5))
    fig_crop.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax_crop = fig_crop.add_subplot(111)
    plt.gray()
    ax_crop.axis('off')
    cmapCenter = np.copy(gt_current)
    crop_offset = 0.20
    ax_crop.set_xlim((cmapCenter[0] - crop_offset, cmapCenter[0] + crop_offset))
    ax_crop.set_ylim((cmapCenter[1] - crop_offset, cmapCenter[1] + crop_offset))
    colors = np.ones(keypoints_projection.shape[1])

    # ax_crop.scatter(keypoints_projection[0,:], keypoints_projection[1,:], \
    #                 s=3, marker='.', c=colors, linewidths=0,\
    #                 label='Number of points: ' + str(keypoints_projection.shape[1]))
    ax_crop.scatter(rawpoints_projection[0,:], rawpoints_projection[1,:], \
                    s=1, marker='.', c='blue', linewidths=0, alpha= 0.8)
                    # label='Number of raw points: ' + str(rawpoints_projection.shape[1]))
    ax_crop.legend()

    # ax_crop.quiver(keypoints_projection[0,:], keypoints_projection[1,:], \
    #                 normals_global[0,:], normals_global[1,:], width=0.0002)

    ax_crop.plot(gt_current[0], gt_current[1], 'r.', markersize=3)  # gt position

    fig_semantics, ax_semantics = crop_display_semantics(fig_crop, ax_crop, mapgraph, \
                                                         way_line_width, cmapCenter, crop_offset)

    # Visualize mapgraph
    SINK_ROAD = 'SINK_ROAD_KEY'
    dispLines = []
    allStreets = dict((vt, n) for (n, vt) in enumerate(mapgraph.nodes_iter()))
    for curr_street in allStreets:
        if curr_street == SINK_ROAD:
            continue
        origin = mapgraph.node[curr_street]['origin']
        terminus = mapgraph.node[curr_street]['terminus']
        dispLines.append(np.vstack((origin, terminus)))
    dx0 = ax_crop.viewLim.width
    dx1 = ax_crop.bbox.width
    sc = (dx1 / dx0) * (72.0 / fig_semantics.dpi)

    # Route network
    # ax_semantics.add_collection(LineCollection(dispLines,transOffset = ax_crop.transData,
    #                                      linewidths=sc*way_line_width,  colors='black'))
    fname = cwd + '/scan2pointCloud/' + 'gtKeypointProjection' + str(frame_id).zfill(4) + 'Lost.png'
    fig_semantics.savefig(fname, dpi=400)
    plt.close("all")

    # -----------------------------------------Visualization method 2---------------------------------------------------
    # crop_display_semantics_utm(frame_id, gps_list, points, mapgraph)

def track_points(options, frame_idx, k_transformed_pointcloud_list):


    filtered_points = np.empty((3,0))
    filtered_normals = np.empty((3,0))

    surface_points_k_frame = []
    normals_k_frame = []
    point_tree_k_frame = []
    width = 1.0
    angle_diff_thres = 0.0872665 # radians
    for j in range(options.data_skip):
        temp_pointcloud = k_transformed_pointcloud_list[j]
        transformed_pointcloud_xy_coordinate_per_frame = np.ones((4, temp_pointcloud.shape[1]))
        # Convert back to our IROS/IJRR coordinate
        transformed_pointcloud_xy_coordinate_per_frame[0, :] =  temp_pointcloud[2, :]
        transformed_pointcloud_xy_coordinate_per_frame[1, :] = -temp_pointcloud[0, :]
        surface_point_per_frame, normals_per_frame = \
            oriented_surface_points(transformed_pointcloud_xy_coordinate_per_frame)
        points_tree = build_rtree_points(surface_point_per_frame)
        surface_points_k_frame.append(surface_point_per_frame)
        normals_k_frame.append(normals_per_frame)
        point_tree_k_frame.append(points_tree)

    k_frame_kept_idxs = []

    for j in range(options.data_skip):
        kept_feature_idxs = []
        angles = np.arctan2(normals_k_frame[j][1, :], normals_k_frame[j][0, :])
        for i in range(surface_points_k_frame[j].shape[1]):
            num_reject_feature = 0
            num_voting_feature = 0

            x = surface_points_k_frame[j][0, i]
            y = surface_points_k_frame[j][1, i]
            angle = angles[i]

            if angle < 0:
                angle = angle + np.pi

            for k in range(options.data_skip):
                if k == j:
                    continue
                intersect_idx = list(point_tree_k_frame[k].intersection((x - width, y - width, \
                                                                         x + width, y + width)))
                per_frame_flag = 0
                for idx in intersect_idx:
                    query_point_angle = np.arctan2(normals_k_frame[k][1, idx], normals_k_frame[k][0, idx])
                    if query_point_angle < 0:
                        query_point_angle = query_point_angle + np.pi
                    if abs(query_point_angle - angle) < angle_diff_thres:
                        per_frame_flag += 1
                    else:
                        per_frame_flag -= 1
                if per_frame_flag > 0:
                    num_voting_feature += 1
                else:
                    num_reject_feature += 1


            if num_voting_feature > 2:
                kept_feature_idxs.append(i)

        k_frame_kept_idxs.append(kept_feature_idxs)
        filtered_points = np.hstack((filtered_points, surface_points_k_frame[j][0:3, kept_feature_idxs]))
        filtered_normals = np.hstack((filtered_normals, normals_k_frame[j][:, kept_feature_idxs]))

    # import matplotlib
    # matplotlib.use('agg')
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111)
    # colors = ['k', 'b', 'g', 'r', 'y']
    #     ax.quiver(surface_points_k_frame[j][0, kept_feature_idxs],\
    #                             surface_points_k_frame[j][1, kept_feature_idxs], \
    #                             np.cos(angles[kept_feature_idxs]), np.sin(angles[kept_feature_idxs]), \
    #                             width=0.002, color=colors[j], alpha=1)
    # ax.set_xlim(-200, 200)
    # ax.set_ylim(-200, 200)
    # directory = cwd + '/scan2pointCloud/perframe_normal' + '_' + str(frame_idx).zfill(6)
    # fig_name = directory + '.png'
    # fig.savefig(fig_name, dpi=200)
    # plt.close('all')

    return filtered_points, filtered_normals

def load_radar_pointcloud(options, start_frame_id):
    print ('Start to load pointcloud......')
    pointcloud_save_path = options.radar_data + '../pointcloud/'
    normals_save_path = options.radar_data + '../normals/'

    # check if the pointcloud already saved in the disk
    if os.path.exists(pointcloud_save_path):
        print ('Reading pointcloud......')

        accumulated_surface_points = []
        surface_points_normals = []
        accumulated_raw_points = [] # not using it ATM

        pointcloud_names = sorted(glob.glob(pointcloud_save_path + '*.csv'))
        normal_names = sorted(glob.glob(normals_save_path + '*.csv'))
     
        for pointcloud_path in pointcloud_names:
            pointcloud = np.loadtxt(pointcloud_path, delimiter=",")
            accumulated_surface_points.append(pointcloud)

        for normal_path in normal_names:    
            normal = np.loadtxt(normal_path, delimiter=",")
            surface_points_normals.append(normal)
    # if not, extract them from the raw radar images
    else:
        accumulated_surface_points, \
        surface_points_normals,\
        accumulated_raw_points = extract_radar_pointcloud(options, start_frame_id)

    return accumulated_surface_points, surface_points_normals, accumulated_raw_points

def extract_radar_pointcloud(options, start_frame_id):
    odometry_list = load_radar_odometry(options) # load odometry

    print ('Extracting pointcloud......')
    # load all pointclouds
    max_range = float(options.max_range)
    selected_max_distance = float(options.selected_max_distance)
    radar_resolution = float(options.radar_resolution)
    image_names = sorted(glob.glob(options.radar_data + '*.png'))
    pointcloud_save_path = options.radar_data + '../pointcloud/'
    if not os.path.exists(pointcloud_save_path):
        os.mkdir(pointcloud_save_path)
    normals_save_path = options.radar_data + '../normals/'
    if not os.path.exists(normals_save_path):
        os.mkdir(normals_save_path)

    numImages = len(image_names)
    k_frame_pointcloud = []
    poses_old = []
    accumulated_surface_points = []
    accumulated_raw_points = []
    surface_points_normals = []
    counter = 0
    counter_data_skip = 0
    for i, img_path in enumerate(image_names):
        if i < start_frame_id - options.data_skip + 1:
            continue
        start_time = time.time()
        img = cv2.imread(img_path,0)
        if 'oxford' in options.dataname or 'boreas' in options.dataname:
            img = img[:, 11:]
        counter += 1

        if i < 2:
            pose_previous = np.eye(4, 4)
            pose_current = np.eye(4, 4)
        else:
            pose_previous = odometry_list[i - 2]
            pose_current = odometry_list[i - 1]

        pointcloud_np = extract_pointcloud(i, img, max_range, selected_max_distance, \
                                           radar_resolution, pose_current, pose_previous)
        k_frame_pointcloud.append(pointcloud_np)

        # # Todo: remove it once not debugging
        # if debugFlag:
        #     if i > 20:
        #         break

        # Todo: for now we are setting all points equal confidence
        # Transform last K frame to current frame coordinate. Point [x; y,; z; confidence]
        # Assign lower confidence to the points which are older.
        if (counter % options.data_skip) == 0:
            start_time_surface = time.time()
            transformed_pointcloud = k_frame_pointcloud[-1]
            k_transformed_pointcloud_list = []
            k_transformed_pointcloud_list.append(transformed_pointcloud)
            for j in range(options.data_skip-1):
                pose_old = poses_old[j]
                transformation_to_current_frame = compute_relative_pose(pose_old, pose_current)
                old_pointcloud = k_frame_pointcloud[j]
                temp_pointcloud = transformation_to_current_frame.dot(old_pointcloud)
                transformed_pointcloud = np.hstack((transformed_pointcloud,temp_pointcloud))
                k_transformed_pointcloud_list.append(transformed_pointcloud)


            filtered_surface_points, filtered_normals = track_points(options, i, k_transformed_pointcloud_list)
            accumulated_surface_points.append(filtered_surface_points)
            surface_points_normals.append(filtered_normals)
            pointcloud_filename = pointcloud_save_path + str(counter_data_skip).zfill(6) + '.csv'
            normal_filename = normals_save_path + str(counter_data_skip).zfill(6) + '.csv'
            np.savetxt(pointcloud_filename, filtered_surface_points, delimiter=",")
            np.savetxt(normal_filename, filtered_normals, delimiter=",")
            # print 'pointcloud shape = {0}, normal shape = {1}'.format(filtered_surface_points.shape, filtered_normals.shape)
            counter_data_skip += 1

            # Convert back to our IROS/IJRR coordinate
            transformed_pointcloud_xy_coordinate = np.ones((4,transformed_pointcloud.shape[1]))
            transformed_pointcloud_xy_coordinate[0,:] = transformed_pointcloud[2,:]
            transformed_pointcloud_xy_coordinate[1,:] = -transformed_pointcloud[0,:]
            accumulated_raw_points.append(transformed_pointcloud_xy_coordinate)

            # reset these variables
            k_frame_pointcloud = []
            poses_old = []
            counter = 0
            # elapsed_time =  time.time() - start_time_surface
            # print 'Generate {0} surface points computation time = {1}'.format(surface_points_xy.shape[1], elapsed_time)
        else:
            poses_old.append(pose_current)
        elapsed_time =  time.time() - start_time
        sys.stdout.write('\r{0:4.3}% ({1:6} of {2} images processed. Each image processed time: {3})'.format(
                (100.0 * i) / numImages, i, numImages, elapsed_time))

    return accumulated_surface_points, surface_points_normals, accumulated_raw_points

def crop_display_semantics(fig, ax, mapgraph, way_line_width, fig_center, offset):
    current_year = 2022.0

    import matplotlib.pyplot as plt
    fig_semantics = plt.figure(dpi = fig.dpi,figsize=fig.get_size_inches())
    ax_semantics = fig_semantics.add_subplot(111)
    ax_semantics.axis('off')
    dx0 = ax.viewLim.width
    dx1 = ax.bbox.width
    sc = (dx1 / dx0) * (72.0 / fig.dpi)
    all_nodes = mapgraph.mapSemantics.nodes_dict
    mercator_scale = mapgraph.graph['mercator_scale']

    line = []
    year_diff_list = []
    min_x_map = min_y_map = np.inf
    max_x_map = max_y_map = -np.inf
    for bd in mapgraph.mapSemantics.buildings:
        polygon = []
        for ndRef in bd.nodes:
            latlon = all_nodes[ndRef.ref]
            pos = mercatorProj(latlon, mercator_scale) - mapgraph.graph['position_offset']
            if pos[0] < min_x_map:
                min_x_map = pos[0]
            if pos[1] > max_y_map:
                max_y_map = pos[1]
            if pos[0] > max_x_map:
                max_x_map = pos[0]
            if pos[1] < min_y_map:
                min_y_map = pos[1]
            polygon.append(pos)
        disPolygon = []
        for i in range(len(polygon)):
            if i > 0:
                disPolygon.append([[polygon[i - 1][0], polygon[i - 1][1]], [polygon[i][0], polygon[i][1]]])
                line.append([[polygon[i - 1][0], polygon[i - 1][1]], [polygon[i][0], polygon[i][1]]])
                year_diff = current_year - bd.timestamp.date().year
                year_diff_list.append(year_diff)


    for pw in mapgraph.mapSemantics.footways:
        positions = []
        # line = []
        for ndRef in pw.nodes:
            latlon = all_nodes[ndRef.ref]
            pos = mercatorProj(latlon, mercator_scale) - mapgraph.graph['position_offset']
            if pos[0] < min_x_map:
                min_x_map = pos[0]
            if pos[1] > max_y_map:
                max_y_map = pos[1]
            if pos[0] > max_x_map:
                max_x_map = pos[0]
            if pos[1] < min_y_map:
                min_y_map = pos[1]
            positions.append(pos)
        for i in range(len(positions) - 1):
            line.append(np.vstack((positions[i], positions[i + 1])))
            year_diff = current_year - pw.timestamp.date().year
            year_diff_list.append(year_diff)


    line_collection = LineCollection(line, transOffset=ax.transData, \
                       linewidths=sc * way_line_width, alpha=0.5)
    line_collection.set_array(np.asarray(year_diff_list))
    ax.add_collection(line_collection)
    axcb = fig.colorbar(line_collection)
    axcb.set_label('Year Difference')

    ax.set_xlim((min_x_map, max_x_map))
    ax.set_ylim((min_y_map, max_y_map))

    fig_hist = plt.figure(dpi = fig.dpi,figsize=fig.get_size_inches())
    ax_hist = fig_hist.add_subplot(111)
    # print max(year_diff_list)
    ax_hist.hist(year_diff_list, bins=int(max(year_diff_list)), density=True)
    ax_hist.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    directory = cwd + '/results/visualization/'
    fig_name = directory +  'map_hist.png'
    fig_hist.savefig(fig_name, dpi=200)

    return fig,ax


def crop_display_semantics_utm(tInd, gps_list, points, mapgraph):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    way_line_width = 0.0005

    fig = plt.figure(dpi=200, figsize=(5, 5))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax = fig.add_subplot(111)
    plt.gray()
    gt_curr_utm = utm.from_latlon(gps_list[tInd][0], gps_list[tInd][1])  # in meters
    gt_prev_utm = utm.from_latlon(gps_list[tInd - 1][0], gps_list[tInd - 1][1])
    x = gt_curr_utm[0] - gt_prev_utm[0]
    y = gt_curr_utm[1] - gt_prev_utm[1]

    heading_mercator = np.arctan2(y, x)
    transformation = np.asarray([[np.cos(heading_mercator), -np.sin(heading_mercator), 0], \
                                 [np.sin(heading_mercator), np.cos(heading_mercator), 0], \
                                 [0, 0, 1]])
    transformed_points = np.dot(transformation, points[0:3, :])
    points_utm = transformed_points[0:2, :] + np.array([[gt_curr_utm[0]], [gt_curr_utm[1]]])

    tmpt = time.time()
    num_points = points_utm.shape[1]
    mercator_visualization_x = []
    mercator_visualization_y = []
    for i in range(num_points):
        pt_utm = points_utm[:, i].transpose()
        mercator_visualization_x.append(pt_utm[0])
        mercator_visualization_y.append(pt_utm[1])

    print 'UTM to GPS conversion for points done in {0}s.'.format(time.time() - tmpt)

    tmpt = time.time()



    ax.axis('off')


    colors = np.ones(len(mercator_visualization_x))

    ax.scatter(mercator_visualization_x, mercator_visualization_y, \
                    s=3, marker='.', c=colors, linewidths=0)



    dx0 = ax.viewLim.width
    dx1 = ax.bbox.width
    sc = (dx1 / dx0) * (72.0 / fig.dpi)

    all_nodes = mapgraph.mapSemantics.nodes_dict


    for bd in mapgraph.mapSemantics.buildings:
        polygon = []
        for ndRef in bd.nodes:
            latlon = all_nodes[ndRef.ref]
            # print nd
            # print latlon
            pos = utm.from_latlon(latlon[0], latlon[1])
            xy =  np.array([pos[0],pos[1]])
            polygon.append(xy)
        disPolygon = []
        for i in range(len(polygon)):
            if i > 0:
                disPolygon.append([[polygon[i - 1][0], polygon[i - 1][1]], [polygon[i][0], polygon[i][1]]])

        # ax.add_collection(LineCollection(disPolygon, transOffset=ax.transData,
        #                                  linewidths=sc * way_line_width, colors='black'))
        #                                     # linewidths=sc * WAY_LINE_WIDTH, colors='purple'))
        ax.add_collection(LineCollection(disPolygon, transOffset=ax.transData,
                                         linewidths=0.1, colors='black', alpha=0.5))

    for pw in mapgraph.mapSemantics.footways:
        positions = []
        line = []
        for ndRef in pw.nodes:
            latlon = all_nodes[ndRef.ref]
            pos = utm.from_latlon(latlon[0], latlon[1])
            xy =  [pos[0],pos[1]]
            positions.append(xy)
        for i in range(len(positions) - 1):
            line.append(np.vstack((positions[i], positions[i + 1])))
        # ax.add_collection(LineCollection(line, transOffset=ax.transData, \
        #                    linewidths=sc * way_line_width, colors='black'))
        ax.add_collection(LineCollection(line, transOffset=ax.transData,\
                                         linewidths=0.1, \
                                         colors='black',alpha=0.5))

    # ax.set_xlim((gt_curr_utm[0] - crop_offset, gt_curr_utm[0] + crop_offset))
    # ax.set_ylim((gt_curr_utm[1] - crop_offset, gt_curr_utm[1] + crop_offset))

    print 'Drawing pointcloud figure done in {0}s.'.format(time.time() - tmpt)
    ax.plot(gt_curr_utm[0], gt_curr_utm[1], 'r.', markersize=1)  # gt position
    ax.axis('off')
    crop_offset = 300
    ax.set_xlim((gt_curr_utm[0] - crop_offset, gt_curr_utm[0] + crop_offset))
    ax.set_ylim((gt_curr_utm[1] - crop_offset, gt_curr_utm[1] + crop_offset))
    fname = cwd + '/scan2pointCloud/' + 'postCropDisplay' + str(tInd + 1).zfill(4) + 'UTM.png'
    # fig_semantics.savefig(fname,bbox_inches='tight',dpi = displayDPI)
    fig.savefig(fname, dpi=200)
    print 'Saving crop figure done in {0}s.'.format(time.time() - tmpt)

def fig2array(fig):
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    array = np.asarray(image)
    return array

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


