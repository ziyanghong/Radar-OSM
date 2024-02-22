import os
cwd = os.getcwd()

# from multiprocessing import Pool
import math
import numpy as np

from matplotlib.collections import LineCollection

from MapGraph import MapGMMDistribution,SINK_ROAD

import time
from RadarUtils import build_rtree_points
from TransformUtils import *
from Statistics import *

# numba
from numba import njit, cuda, prange, vectorize, float64
from numba.typed import List



# ##for cythonizing my code
# from cythonized import projection_point_and_distances
# import cProfile
# import pstats
# import StringIO

# for rtree
from rtree import index


SMALL_COMPONENT_THRESH = 0.05
MAX_PROJECT_DIST = 0.002 # km orginal setting
NORMAL_DIFF_THRESH = 0.35  # radians
LANE_WIDTH = 3.75 / 1000 # km
LANE_ERROR_OXFORD = 3.0 / 1000 # KM
LANE_ERROR_MULRAN = 3.0 / 1000
LANE_ERROR_BOREAS = 2.0 / 1000
CYCLE_WAY_WIDTH = 1.25 / 1000
ONE_WAY_WIDTH = 0.016 # km * project_scale (0.8)

moveVarX = 0.0004 **2 # variance on x (km)
moveVarY = 0.00017 **2 # variance on y (km)
moveVarAngle = 0.017 ** 2# variance on angle (radian)

dv_offset_move_sigma = 0.0003 # m/s
longitudinal_num_correction_thresh = 0.6
obsSigma_x = 0.05 **2 # unit in km
obsSigma_y = 0.0008 **2 # unit in km
obsSigma_angle  = 0.1 **2
heading_thres = 60.0 / 180.0 * np.pi
calibrationErrorX = 0.027 / 1000.0
calibrationErrorY = -0.110583515418169 / 1000.0
calibrationErrorAngle = 0.00018

# A mode is kept if streetMarginThresh percent of its probability mass is inside [-streetMargin,len+streetMargin]
streetMargin = 0
streetMarginThresh = 0.001
# To draw
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
displayOffset = 0.08  # km
displayFigSz = (7,7)
displayDPI = 100
drawPosterior = True
drawObservation = True
drawFeatures = True
debugFilter = False
debugPredictModel = False
debugFrame = 2880


def propagate_uncertainty(xk, uk, Pk_1, Qk):
    """

    Input:
        xk: previous state, 1 x 3 vector
        uk: odometry, 1 x 3 vector
        Pk_1: previous uncertainty, 3 x 3 matrix
        Qk: process noise
    Output:
        Pk: updated covariance

    """
    Ak = np.array([[1.0 ,0.0, - math.sin(xk[2]) * uk[0] - math.cos(xk[2]) * uk[1]],\
                   [0.0 ,1.0,   math.cos(xk[2]) * uk[0] - math.sin(xk[2]) * uk[1]],\
                   [0.0 ,0.0,                                                 1.0]])
    Wk = np.array([[math.cos(xk[2]), - math.sin(xk[2]), 0.0],\
                   [math.sin(xk[2]),   math.cos(xk[2]), 0.0],\
                   [0.0,        0.0,                    1.0]])

    P1 = np.dot(Ak, Pk_1)
    P1 = np.dot(P1, np.transpose(Ak))
    P2 = np.dot(Wk, Qk)
    P2 = np.dot(P2, np.transpose(Wk))

    # Predict the uncertainty
    Pk = P1 + P2
    return Pk


@vectorize([float64(float64)])
def wrap_angle(angle_radian):
    # To wrap angle from -pi/2 to pi/2
    if angle_radian > (np.pi / 2):
        wrapped_angle_radian = angle_radian - np.pi
    elif angle_radian < (-np.pi / 2):
        wrapped_angle_radian = angle_radian + np.pi
    else:
        wrapped_angle_radian = angle_radian
    return wrapped_angle_radian


def filter_segments(intersect_segments, predicted_pos, street_angle, \
    intersection_offset):
    # print 'street angle = {0}'.format(street_angle/np.pi*360)

    # Remove some segments base on distance and reflectivity
    number_of_angles = 240
    CLOSE_OBJ_DIST_THRES = 0.1 # km
    NUM_DETECTION_PER_BEAM = 4
    SEGMENT_ANGLE_DIFF_THRESH = 60.0 / 180 * np.pi

    new_intersect_idx = []



    curr_x = predicted_pos[0]
    curr_y = predicted_pos[1]
    angle_resolution = np.arange(-np.pi, np.pi, np.pi*2/number_of_angles)
    # print angle_resolution
    angles_dist_list = dict()
    segment_angle_dict = dict()
    segment_dist_dict = dict()
    for i in range(number_of_angles):
        angles_dist_list[i] = []

    for l in range(intersect_segments.shape[0]):
        line = intersect_segments[l,:]
        line_mid_x = (line[2] + line[0])/2
        line_mid_y = (line[3] + line[1])/2
        dist = np.sqrt((line_mid_y - curr_y)**2 + (line_mid_x - curr_x)**2)
        angle = np.arctan2(line_mid_y - curr_y, line_mid_x - curr_x)
        index = np.abs(angle_resolution - angle).argmin()
        # closest_angle = angle_resolution[index]
        angles_dist_list[index].append(dist)
        segment_angle_dict[l] = index
        segment_dist_dict[l] = dist

    for l in range(intersect_segments.shape[0]):
        line = intersect_segments[l,:]
        c1 = np.sqrt((curr_x - line[0])**2 + (curr_y - line[1])**2) > intersection_offset
        c2 = np.sqrt((curr_x - line[2])**2 + (curr_y - line[3])**2) > intersection_offset
        if c1 and c2:
            continue
        sorted_dist = sorted(angles_dist_list[segment_angle_dict[l]])
        c3 = sorted_dist.index(segment_dist_dict[l]) >= NUM_DETECTION_PER_BEAM
        if c3:
            continue

        c4 = sorted_dist.index(segment_dist_dict[l]) >= (NUM_DETECTION_PER_BEAM - 2)    
        seg_2_street_angle = angle_resolution[segment_angle_dict[l]]
        angle_diff = wrap_angle( abs(seg_2_street_angle - street_angle)) 
        # if segment is orthogonal to street
        c5 = SEGMENT_ANGLE_DIFF_THRESH < angle_diff < (np.pi - SEGMENT_ANGLE_DIFF_THRESH) 
        if c4 and c5:
            continue

        new_intersect_idx.append(l)
    filtered_intersect_segments = intersect_segments[np.asarray(new_intersect_idx)] 

    return filtered_intersect_segments

def evaluate_associations(radar_points_local, radar_normal_local, radar_point_global, map_point_global, associations):
    bool_longitudinal_correction = False
    bool_lateral_correction = False
    longitudinal_correction_feature_points = []
    lateral_correction_feature_points = []
    lateral_errors = []
    long_features_idx = []
    lateral_features_idx = []
    num_left_features = 0
    num_right_features = 0

    # Thresholds
    num_correction_thresh = 30
    num_weak_side_thresh = 10
    lateral_spatial_thresh = 0.08 # km
    long_c2_std_x = 0.05 # km
    long_c3_std_y = 0.03
    normal_threshold = 30.0 / 180.0 * np.pi
    mean_points_residual_thresh = 0.005

    # Check each association, assign it to either lateral or longitudinal feature
    for i in range(associations.shape[0]):
        radar_pt_idx = associations[i,0]
        map_pt_idx = associations[i,1]
        wrap_angle_normal = wrap_angle(radar_normal_local[radar_pt_idx])
        # print wrap_angle_normal
        pt_local = radar_points_local[0:2, radar_pt_idx]

        if abs(wrap_angle_normal) < normal_threshold:
            longitudinal_correction_feature_points.append(pt_local)
            long_features_idx.append(radar_pt_idx)
        elif abs(wrap_angle_normal) > (np.pi/2 - normal_threshold) :
            lateral_correction_feature_points.append(pt_local)
            lateral_features_idx.append(radar_pt_idx)
            delta_x = radar_point_global[0, radar_pt_idx] - map_point_global[0, map_pt_idx]
            delta_y = radar_point_global[1, radar_pt_idx] - map_point_global[1, map_pt_idx]
            lateral_errors.append(np.sqrt(delta_x**2 + delta_y**2))
            if pt_local[1] > 0.0:
                num_left_features += 1
            else:
                num_right_features += 1

    lateral_c1 = len(lateral_correction_feature_points) > num_correction_thresh
    if lateral_c1:
        lateral_points_cov = np.cov(np.asarray(lateral_correction_feature_points).T)
        # print 'points distribution lateral covariance = {0}'.format(np.sqrt(lateral_points_cov[1,1]))
        lateral_c2 = np.sqrt(lateral_points_cov[1,1]) > lateral_spatial_thresh **2
    else:
        lateral_c2 = False

    # Criteria 3: this might induce by map error
    lateral_c3 = np.mean(np.asarray(lateral_errors)) < mean_points_residual_thresh
    lateral_std = np.std(np.asarray(lateral_errors))
    # print 'lateral points error std = {0}'.format(lateral_std)

    # Criteria 4: check left and right feature imbalance
    num_weak_side = min(num_left_features, num_right_features)
    if num_weak_side < num_weak_side_thresh:
        lateral_c4 = False
    else:
        lateral_c4 = True



    if lateral_c1 and lateral_c2 and lateral_c4:
        bool_lateral_correction = True

    num_long_points = len(longitudinal_correction_feature_points)
    # print 'num_long_points = {0}'.format(num_long_points)
    long_c1 = num_long_points > num_correction_thresh
    if long_c1:
        long_points_cov = np.cov(np.asarray(longitudinal_correction_feature_points).T)
        long_c2 = np.sqrt(long_points_cov[0,0]) > long_c2_std_x
        long_c3 = np.sqrt(long_points_cov[1,1]) > long_c3_std_y
        # print 'long_x = {0}, long_y = {1}.'.format(np.sqrt(long_points_cov[0, 0]), np.sqrt(long_points_cov[1, 1]))
    else:
        long_c2 = False
        long_c3 = False

    # if long_c1 and long_c2 and long_c3:
    #     bool_longitudinal_correction = True
    if long_c1 and long_c3:
        bool_longitudinal_correction = True

    # print 'long_c1 = {0}, long_c2 = {1}, long_c3 = {2}, lat_c1 = {3}, lat_c2 = {4}, lat_c4 = {5}.'.format(long_c1,\
    #         long_c2, long_c3, lateral_c1, lateral_c2, lateral_c4)
    return bool_longitudinal_correction, bool_lateral_correction, long_features_idx, lateral_features_idx

# @njit
# def projection_point_and_distances(local_radar_points, poses_SE2, project_mat, num_intersect_segments):
#     """
#     Input:
#         local_radar_points: 4 x N
#         poses_SE2: 3 x 3 x P
#         project_mat:  2 x 3 x L
#         num_intersect_segments: L
#     Output:
#         projections: P x L x 2 x N
#         distance_mat: P x L x N
#     """
#     distance_mat = np.zeros((poses_SE2.shape[2], num_intersect_segments, local_radar_points.shape[1]))
#     projections = np.zeros((poses_SE2.shape[2], num_intersect_segments, 2, local_radar_points.shape[1]))
#     local_points_x_y_1 = local_radar_points[0:3, :]
#     for p in range(poses_SE2.shape[2]):
#         global_points_x0_y0_1 = np.dot(poses_SE2[:,:,p], local_points_x_y_1)

#         for l in range(num_intersect_segments):
#             m_square = project_mat[1,0,l]**2
#             projection_points = np.dot(project_mat[:,:,l], global_points_x0_y0_1) / (1 + m_square)
#             x0_minus_xp = np.add(projection_points[0, :],  -global_points_x0_y0_1[0, :])
#             y0_minus_yp = np.add(projection_points[1, :],  -global_points_x0_y0_1[1, :])
#             distances = np.sqrt(x0_minus_xp**2 + y0_minus_yp**2)
#             distance_mat[p,l,:] = distances
#             projections[p, l, :, :] = projection_points


#     return projections, distance_mat




def color_title(labels, colors, textprops={'size': 'large'}, ax=None, y=1.013,
                precision=10 ** -2):
    "Creates a centered title with multiple colors."

    if ax == None:
        ax = plt.gca()

    plt.gcf().canvas.draw()
    transform = ax.transAxes  # use axes coords

    # initial params
    xT = 0  # where the text ends in x-axis coords
    shift = 0  # where the text starts

    # for text objects
    text = dict()

    while (np.abs(shift - (1 - xT)) > precision) and (shift <= xT):
        x_pos = shift

        for label, col in zip(labels, colors):

            try:
                text[label].remove()
            except KeyError:
                pass

            text[label] = ax.text(x_pos, y, label,
                                  transform=transform,
                                  ha='left',
                                  color=col,
                                  **textprops)

            x_pos = text[label].get_window_extent(renderer=fig.canvas.get_renderer()) \
                .transformed(transform.inverted()).x1

        xT = x_pos  # where all text ends

        shift += precision / 2  # increase for next iteration

        if x_pos > 1:  # guardrail
            break

def debug_visualize_global_observation(options, radar_global, map_global, local_radar_normals_radian, top_pose, intersect_segments, \
                            associations, gt_pos, frame_idx, street_id, component_id, b_long_temp, b_lateral_temp,\
                                       best_long_features_idx, best_lateral_features_idx, mapgraph, street):

    # To debug normals angle and point angle
    visualization_offset = 0.2  # km
    # import matplotlib
    # matplotlib.use('agg')
    # import matplotlib.pyplot as plt



    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    points_normal_global = np.zeros(radar_global.shape[1])
    for i in range(radar_global.shape[1]):
        point_normal_global_angle = wrap_angle(local_radar_normals_radian[i] + top_pose[2])
        points_normal_global[i] = point_normal_global_angle

    lines = []
    for l_idx in range(intersect_segments.shape[0]):
        line = intersect_segments[l_idx, :]
        lines.append(np.vstack((line[0:2], line[2:4])))

    ax.add_collection(
        LineCollection(lines, transOffset=ax.transData, linewidths=0.1, colors='green'))
    ax.quiver(radar_global[0, associations[:,0]],  radar_global[1, associations[:,0]],\
              np.cos(points_normal_global[associations[:,0]]), np.sin(points_normal_global[associations[:,0]]), \
              width=0.002, color='blue', alpha=0.4)
    ax.scatter(map_global[0, associations[:,1]], map_global[1, associations[:,1]], c='green', s=0.02)

    if len(best_long_features_idx) > 0:
        # To show lateral and long features
        # ax.scatter(radar_global[0, best_long_features_idx], radar_global[1, best_long_features_idx], alpha=0.4, \
        #            s=0.2, c='purple', label='long features')
        ax.quiver(radar_global[0, best_long_features_idx],  radar_global[1, best_long_features_idx],\
              np.cos(points_normal_global[best_long_features_idx]), np.sin(points_normal_global[best_long_features_idx]), \
              width=0.002, color='purple', alpha=0.4, label=' long features')
        
    if len(best_lateral_features_idx) > 0:
        ax.quiver(radar_global[0, best_lateral_features_idx],  radar_global[1, best_lateral_features_idx],\
              np.cos(points_normal_global[best_lateral_features_idx]), np.sin(points_normal_global[best_lateral_features_idx]), \
              width=0.002, color='yellow', alpha=0.4, label=' lateral features')


    # titles_list = ['frame = ' + str(frame_idx) + ', street id = ' + str(street_id) + ' ' + ', comp id = ' + str(component_id) + \
    #                '\nLongitude correction = ',  str(b_long_temp), ', lateral correction = ', str(b_lateral_temp)]
    # if b_long_temp:
    #     long_color = 'green'
    # else:
    #     long_color = 'red'
    # if b_lateral_temp:
    #     lat_color = 'green'
    # else:
    #     lat_color = 'red'
    # tt_colors = ['black', long_color, 'black', lat_color]
    # color_title(titles_list, tt_colors)

    ax.scatter(gt_pos[0], gt_pos[1], s=0.2, c='red')
    ax.quiver(top_pose[0],  top_pose[1],\
               np.cos(top_pose[2]), np.sin(top_pose[2]), width=0.002, color='black')
    ax.set_xlim([gt_pos[0] - visualization_offset, gt_pos[0] + visualization_offset])
    ax.set_ylim([gt_pos[1] - visualization_offset, gt_pos[1] + visualization_offset])
    ax.legend(loc='lower left')
    ax.set_title('Frame = ' + str(frame_idx) + '\nLong correction = ' + str(b_long_temp) + ', lat correction = ' + str(b_lateral_temp))

    ax.plot([mapgraph.node[street]['origin'][0], mapgraph.node[street]['terminus'][0]], \
            [mapgraph.node[street]['origin'][1], mapgraph.node[street]['terminus'][1]], \
            linewidth=0.3, c='black')
    directory = cwd + '/results/visualization/' + options.dataname + \
                '/debug_point_line_normal_' + str(frame_idx).zfill(4) 
    fig_name = directory +'.png'
    fig.savefig(fig_name, dpi=displayDPI)
    plt.close()

def associate_points_with_one_pose(input_list):
    local_map_rtree = input_list[0] 
    global_points_x0_y0_1 = input_list[1]
# def associate_points_with_one_pose(local_map_rtree, global_points_x0_y0_1):
    num_associations = 0
    temp_point_line_dic = {}
    for n in range(num_points):
        global_point = global_points_x0_y0_1[:,n]
        lines_in_box_idx = list(local_map_rtree.intersection((global_point[0]-association_dist_thresh,\
                                                              global_point[1]-association_dist_thresh,\
                                                              global_point[0]+association_dist_thresh,\
                                                              global_point[1]+association_dist_thresh)))

        point_normal_global_angle = wrap_angle(local_radar_normals_radian[n] + pose[2])
        point_lines_list = []
        for line_idx in lines_in_box_idx:
            line_normal_global_angle = wrap_angle(lines_normal_angle[line_idx])
            normal_angle_diff = abs(line_normal_global_angle - point_normal_global_angle)
            # print 'Angle diff = {0}'.format(normal_angle_diff)
            if normal_angle_diff < NORMAL_DIFF_THRESH:
                point_lines_list.append(line_idx)
                num_associations += 1
            temp_point_line_dic[n] = point_lines_list
    results = []
    results.append(temp_point_line_dic)
    results.append(num_associations)
    return results

def find_associations_from_map_rtree(local_radar_points, local_radar_normals_radian, poses, intersect_segments, \
                               frame_idx, street_id, component_id, gt_pos):
    """
    # after submission version 2

    Imput:
        radar_points: 4 x N matrix [x, y, z, 1]^T, radar points
        poses: N x3 matrix [x, y, rotation]

    Output:
        D1:   3 x N matrix [x, y, z], local radar points
        D2 :  3 x M matrix, sample points from map semantics

    """
    start_time = time.time()

    visualization_offset = 0.2
    association_dist_thresh = MAX_PROJECT_DIST
    one_to_n_matches = []
    pose_indexes = []
    poses_points_correspondence = []
    poses_SE2 = np.zeros(( 3, 3, poses.shape[0]))
    for i in range(poses.shape[0]):
        poses_SE2[:,:,i] = np.asarray([[np.cos(poses[i,2]), -np.sin(poses[i,2]), poses[i,0]], \
                                       [np.sin(poses[i,2]),  np.cos(poses[i,2]), poses[i,1]], \
                                       [                 0,                   0,          1]])
        poses_points_correspondence.append([])

    lines_normal_angle = np.zeros((intersect_segments.shape[0]))
    local_map_rtree = index.Index()
    # tmpt = time.time()
    for l in range(intersect_segments.shape[0]):
        # compute line normal angle
        line = intersect_segments[l,:]
        m = (line[3] - line[1]) / (line[2] - line[0])
        line_angle = np.arctan(m)
        lines_normal_angle[l] = wrap_angle(line_angle + np.pi/2)
        xmax = max(line[0], line[2])
        xmin = min(line[0], line[2])
        ymax = max(line[1], line[3])
        ymin = min(line[1], line[3])
        local_map_rtree.insert(l, (xmin,ymin,xmax,ymax))
    print 'build tree done in {0}s.'.format(time.time() - start_time)

    # Check based on normal
    D1_global =[]
    D2 =[]
    most_probable_pose_idx = -1
    top_global_pose = []
    local_points_x_y_1 = local_radar_points[0:3,:]
    num_points = local_radar_points.shape[1]
    max_num_associations = 0
    point_line_dic = {}
    # pool = Pool(12)
    for i in range(poses.shape[0]):
        global_pose = poses_SE2[:,:,i]
        pose = poses[i, :]
        global_points_x0_y0_1 = np.dot(global_pose, local_points_x_y_1)
        num_associations = 0
        temp_point_line_dic = {}
        for n in range(num_points):
            global_point = global_points_x0_y0_1[:,n]
            # start_time_inter = time.time() 
            lines_in_box_idx = list(local_map_rtree.intersection((global_point[0]-association_dist_thresh,\
                                                                  global_point[1]-association_dist_thresh,\
                                                                  global_point[0]+association_dist_thresh,\
                                                                  global_point[1]+association_dist_thresh)))
            # print 'intersect done in {0}s.'.format(time.time() - start_time_inter)

            # point_normal_global_angle = wrap_angle(local_radar_normals_radian[n] + pose[2])
            # point_lines_list = []
            # for line_idx in lines_in_box_idx:
            #     line_normal_global_angle = wrap_angle(lines_normal_angle[line_idx])
            #     normal_angle_diff = abs(line_normal_global_angle - point_normal_global_angle)
            #     # print 'Angle diff = {0}'.format(normal_angle_diff)
            #     if normal_angle_diff < NORMAL_DIFF_THRESH:
            #         point_lines_list.append(line_idx)
            #         num_associations += 1
            # temp_point_line_dic[n] = point_lines_list
        # input_list = []
        # input_list.append(local_map_rtree)
        # input_list.append(global_points_x0_y0_1)
        # results = pool.map(associate_points_with_one_pose,input_list)
        # temp_point_line_dic = results[0]
        # num_associations = results[1]
        if num_associations > max_num_associations and num_associations > 0:
            most_probable_pose_idx = i
            top_global_pose = global_pose
            D1_global = global_points_x0_y0_1
            point_line_dic = temp_point_line_dic
            max_num_associations = num_associations


    # compute projection point and association
    if most_probable_pose_idx != -1:
        idx = 0
        associations = np.zeros((max_num_associations, 2))
        D2 = np.zeros((3, max_num_associations))
        for n in range(num_points):
            for line_idx in point_line_dic[n]:
                line = intersect_segments[line_idx,:]
                point_x0_y0_1_global = global_points_x0_y0_1[:,n].reshape((3,1))
                # Use y = mx + b to parameterize the line
                m = (line[3] - line[1]) / (line[2] - line[0])
                m_square = m**2
                b = line[3] - m * line[2]
                project_mat = np.array([[1,      m, -m * b],
                                        [m, m ** 2,      b]])
                # print point_x0_y0_1_global.shape
                project_point = np.dot(project_mat, point_x0_y0_1_global) / (1 + m_square)
                # print project_point.shape

                # project_on_segment = (min(x1, x2) <= x3 <= max(x1, x2)) and (min(y1, y2) <= y3 <= max(y1, y2))
                # if not project_on_segment:
                #     continue

                radar_pt_idx_map_pt_idx = np.array([n, idx])
                associations[idx,:] = radar_pt_idx_map_pt_idx
                D2[:, idx] = np.array([project_point[0,0], project_point[1,0], 1])
                idx += 1 
        associations = associations.astype(int)

    else:
        associations = np.empty(0)

    # print 'find_associations_from_map done in {0}s.'.format(time.time() - start_time)

    return D1_global, D2, associations, top_global_pose, most_probable_pose_idx


def find_associations_from_map(local_radar_points, local_radar_normals_radian, poses, intersect_segments, \
                               frame_idx, street_id, component_id, gt_pos):
    """
    # after submission version using cython or numba speedup

    Imput:
        radar_points: 4 x N matrix [x, y, z, 1]^T, radar points
        poses: N x3 matrix [x, y, rotation]

    Output:
        D1:   3 x N matrix [x, y, z], local radar points
        D2 :  3 x M matrix, sample from map semantics
        associations: K x 2 matrix

    """
    start_time = time.time()
    visualization_offset = 0.2
    association_dist_thresh = MAX_PROJECT_DIST
    num_intersect_segments = intersect_segments.shape[0]

    poses_SE2 = np.zeros(( 3, 3, poses.shape[0]))
    for i in range(poses.shape[0]):
        poses_SE2[:,:,i] = np.asarray([[np.cos(poses[i,2]), -np.sin(poses[i,2]), poses[i,0]], \
                                       [np.sin(poses[i,2]),  np.cos(poses[i,2]), poses[i,1]], \
                                       [                 0,                   0,          1]])


    project_mat = np.zeros((2, 3, num_intersect_segments))
    lines_normal_angle = np.zeros((num_intersect_segments))
    for l in range(num_intersect_segments):
        line = intersect_segments[l,:]
        # Use y = mx + b to parameterize the line
        m = (line[3] - line[1]) / (line[2] - line[0])
        line_angle = np.arctan(m)
        lines_normal_angle[l] = wrap_angle(line_angle + np.pi/2)
        b = line[3] - m * line[2]
        project_mat[:,:,l] = np.asarray([[1,      m, -m * b],
                                         [m, m ** 2,      b]])

    # print 'Compute poses_SE2 and project_mat done in {0}s.'.format(time.time() - start_time)

    ########################## Compute distance matrix ##############################'))
    # print("\033[92m {}\033[00m".format(
    #         '######################### Compute distance matrix ##############################'))
    # projection_point_and_distances_numba_time = time.time()
    # print 'Num of poses = {0}, Num of points = {1}, Num of lines = {2}.'.format(poses.shape[0],\
    #         local_radar_points.shape[1], num_intersect_segments)
    # projections, distance_mat = projection_point_and_distances(local_radar_points[0:3, :], poses_SE2, \
    #                                                           project_mat, num_intersect_segments)
    projections, distance_mat = projection_point_and_distances_numba(local_radar_points[0:3, :], poses_SE2, \
                                                               project_mat, num_intersect_segments)
    # print 'Compute distance matrix done in {0}s.'.format(time.time() - projection_point_and_distances_numba_time)
    # print("\033[92m {}\033[00m".format(
    #         '#########################################################################'))

    # # # Check based on normal
    # D1 = local_radar_points[0:3,:]
    # pose_indexes = []
    # num_points = local_radar_points.shape[1]
    # counter = 0
    # D2 = []
    # one_to_n_matches = []
    # map_points_normal_angle = []
    # num_points = local_radar_points.shape[1]
    # for n in range(num_points):
    #     per_point_to_lines_dist = distance_mat[:,:,n]
    #     poses_lines_idxs = np.argwhere(per_point_to_lines_dist < association_dist_thresh)
    #     # print ''
    #     # print 'point {0} has {1} potential associations'.format(n, poses_lines_idxs.shape[0])
    #     for i in range(poses_lines_idxs.shape[0]):
    #         pose_idx = poses_lines_idxs[i,0]
    #         line_idx = poses_lines_idxs[i,1]
    #         project_point = np.asarray([projections[pose_idx, line_idx, n], projections[pose_idx, line_idx, n + num_points]]) 
    #         line = intersect_segments[line_idx]
    #         x1 = line[0]
    #         y1 = line[1]
    #         x2 = line[2]
    #         y2 = line[3]
    #         x3 = project_point[0]
    #         y3 = project_point[1]
    #         project_on_segment = (min(x1, x2) <= x3 <= max(x1, x2)) and (min(y1, y2) <= y3 <= max(y1, y2))
    #         if not project_on_segment:
    #             continue
   
    #         pose = poses[pose_idx, :]
    #         line_normal_global_angle = wrap_angle(lines_normal_angle[line_idx])
    #         point_normal_global_angle = wrap_angle(local_radar_normals_radian[n] + pose[2])
    #         normal_angle_diff = abs(line_normal_global_angle - point_normal_global_angle)

    #         if debugFilter:
    #             print 'Angle diff = {0}'.format(normal_angle_diff)


    #         if normal_angle_diff < NORMAL_DIFF_THRESH:
    #             # print '--------------------------------One match------------------------------'
    #             one_to_n_matches.append([n, counter])
    #             map_points_normal_angle.append(line_normal_global_angle)
    #             poses_points_correspondence[pose_idx].append(counter)
    #             D2.append(project_point)
    #             pose_indexes.append(pose_idx)
    #             counter += 1
    check_normal_time = time.time()
    num_points = local_radar_points.shape[1]
    one_to_n_matches,\
    pose_indexes, D2,\
    poses_points_correspondence,\
    max_num_associations = check_normal(num_points, projections, distance_mat, \
        association_dist_thresh, intersect_segments, poses, lines_normal_angle, local_radar_normals_radian)
    # print 'Check normal done in {0}s.'.format(time.time() - check_normal_time)

    # print 'Number of one_to_n_matches = {0}'.format(one_to_n_matches.shape[0])
    if max_num_associations > 0:
        # D2 = np.asarray(D2)
        D2 = np.transpose(np.hstack(( D2 , np.ones((D2.shape[0], 1)))))
        associations = one_to_n_matches
        # print associations
        most_probable_pose_idx = pose_indexes[0]
        # associations = np.empty((0, 2))
        # one_to_n_matches = np.asarray(one_to_n_matches)
        # pose_indexes = np.asarray(pose_indexes)
        # most_probable_pose_idx = np.bincount(pose_indexes).argmax()
        # for map_point_idx in poses_points_correspondence[most_probable_pose_idx]:
        #     associations = np.vstack((associations , one_to_n_matches[map_point_idx,:]))

        # for pose_id_point_id in poses_points_correspondence:
        # for i in range(poses_points_correspondence.shape[0]):
        #     pose_id_point_id = poses_points_correspondence[i,:]

        #     if pose_id_point_id[0] == most_probable_pose_idx:
        #         map_point_idx = pose_id_point_id[1]
        #         associations = np.vstack((associations , one_to_n_matches[map_point_idx,:]))
        # print 'Number of initial associations = {0}'.format(associations.shape[0])
        D1_global = np.dot(poses_SE2[:, :, most_probable_pose_idx], local_radar_points[0:3,:])
        # associations = associations.astype(int)
        top_global_pose = poses[most_probable_pose_idx,:]
    else:
        D1_global =[]
        D2 =[]
        associations = np.empty(0)
        top_global_pose = []
        most_probable_pose_idx = -1
    # print 'find_associations_from_map done in {0}s.'.format(time.time() - start_time)

    return D1_global, D2, associations, top_global_pose, most_probable_pose_idx


@njit(parallel=True, fastmath=True)
def numba_func(projections, distance_mat, local_points_x_y_1, poses_SE2, project_mat, m_square_array, num_intersect_segments, num_pose, num_point):

    for l in prange(num_intersect_segments):
        m_square = m_square_array[l]
        inv_1_plus_m_square = 1.0 / (1.0 + m_square)
        project_mat_one_line = project_mat[:,:,l]
        for p in range(num_pose):
            SE2 = poses_SE2[:,:,p]
            for i in range(num_point):
                # global_point_x0 = poses_SE2[0,0,p] * local_points_x_y_1[0,i] + poses_SE2[0,1,p] * local_points_x_y_1[1,i] + poses_SE2[0,2,p]
                # global_point_y0 = poses_SE2[1,0,p] * local_points_x_y_1[0,i] + poses_SE2[1,1,p] * local_points_x_y_1[1,i] + poses_SE2[1,2,p]
                global_point_x0 = SE2[0,0] * local_points_x_y_1[0,i] + SE2[0,1] * local_points_x_y_1[1,i] + SE2[0,2]
                global_point_y0 = SE2[1,0] * local_points_x_y_1[0,i] + SE2[1,1] * local_points_x_y_1[1,i] + SE2[1,2]

                projection_point_x = (project_mat_one_line[0,0] * global_point_x0 + project_mat_one_line[0,1] * global_point_y0 + project_mat_one_line[0,2]) * inv_1_plus_m_square
                projection_point_y = (project_mat_one_line[1,0] * global_point_x0 + project_mat_one_line[1,1] * global_point_y0 + project_mat_one_line[1,2]) * inv_1_plus_m_square
                projections[p, l, i] = projection_point_x
                projections[p, l, i + num_point] = projection_point_y
                x0_minus_xp = projection_point_x - global_point_x0
                y0_minus_yp = projection_point_y - global_point_y0
                distance_mat[p,l,i] = math.sqrt(x0_minus_xp**2 + y0_minus_yp**2)    
    return projections, distance_mat

def projection_point_and_distances_numba(local_points_x_y_1, poses_SE2, project_mat, num_intersect_segments ):
    """
    Input:
        local_points_x_y_1: 3 x N
        poses_SE2: 3 x 3 x P
        project_mat:  2 x 3 x L
        num_intersect_segments: L
    Output:
        projections: P x L x 2N
        distance_mat: P x L x N
    """
    num_pose = poses_SE2.shape[2]
    num_point = local_points_x_y_1.shape[1]
    projections = np.zeros((num_pose, num_intersect_segments, 2*num_point),  dtype= np.double)
    distance_mat = np.ones((num_pose, num_intersect_segments, num_point),  dtype= np.double) 

    m_square_array = project_mat[1,0,:]**2
    projections, distance_mat = numba_func(projections, distance_mat, local_points_x_y_1, poses_SE2, project_mat, m_square_array, num_intersect_segments, num_pose, num_point)


    return projections, distance_mat



def check_normal(num_points, projections, distance_mat, association_dist_thresh, intersect_segments, poses, lines_normal_angle, local_radar_normals_radian):
    pre_allocate_num = 1000

    # output variable 

    # one_to_n_matches = List() 
    # D2 = List()  
    # pose_indexes = List() 
    # poses_points_correspondence = List()   

    one_to_n_matches = np.zeros((pre_allocate_num,2), dtype=int)
    pose_indexes = np.ones((pre_allocate_num), dtype=int)
    D2 = np.zeros((pre_allocate_num, 2), dtype=float)
    poses_points_correspondence = np.zeros((pre_allocate_num,2), dtype=int)
    mask = np.zeros((distance_mat.shape[0],distance_mat.shape[1], distance_mat.shape[2]), dtype=int)
    # for i in range(poses.shape[0]):
    #     poses_points_correspondence.append(np.empty(0))


    one_to_n_matches, pose_indexes,\
     D2, poses_points_correspondence,\
     max_num_associations = numba_func_normal(one_to_n_matches, \
        pose_indexes, D2, poses_points_correspondence, mask, \
        num_points, projections, distance_mat, association_dist_thresh,\
         intersect_segments, poses, lines_normal_angle, local_radar_normals_radian)
    

    return one_to_n_matches, pose_indexes, D2, poses_points_correspondence, max_num_associations

@njit(parallel=True)
def numba_func_normal(one_to_n_matches, pose_indexes, D2, poses_points_correspondence, mask,\
    num_points, projections, distance_mat, association_dist_thresh, intersect_segments, poses,\
     lines_normal_angle, local_radar_normals_radian):
    # counter = 0
    poses_associtaion_counter = np.zeros((poses.shape[0]))
    # poses_associtaion_counter = dict()
    for p in range(poses.shape[0]):
        poses_associtaion_counter[p] = 0

    for n in prange(num_points):
        per_point_to_lines_dist = distance_mat[:,:,n]
        poses_lines_idxs = np.where(per_point_to_lines_dist < association_dist_thresh)
        # print 'number of matches found for this point = {0}'.format(poses_lines_idxs[0].shape[0])

        for i in range(poses_lines_idxs[0].shape[0]):
            # pose_idx = poses_lines_idxs[i,0]
            # line_idx = poses_lines_idxs[i,1]
            pose_idx = poses_lines_idxs[0][i]
            line_idx = poses_lines_idxs[1][i]
            line = intersect_segments[line_idx]
            pose = poses[pose_idx, :]
            line_normal_global_angle = wrap_angle(lines_normal_angle[line_idx])
            point_normal_global_angle = wrap_angle(local_radar_normals_radian[n] + pose[2])
            normal_angle_diff = abs(line_normal_global_angle - point_normal_global_angle)
            # if debugFilter:
            #     print 'Angle diff = {0}'.format(normal_angle_diff)

            condition_normal = normal_angle_diff > NORMAL_DIFF_THRESH
            if condition_normal:
                continue

            project_point = np.asarray([projections[pose_idx, line_idx, n], \
                                        projections[pose_idx, line_idx, n + num_points]]) 
            # x1 = line[0]
            # y1 = line[1]
            # x2 = line[2]
            # y2 = line[3]
            # x3 = project_point[0]
            # y3 = project_point[1]
            # project_on_segment = (min(x1, x2) <= x3 <= max(x1, x2)) and (min(y1, y2) <= y3 <= max(y1, y2))
            bool_x_on_segment = (min(line[0], line[2]) <= project_point[0] <= max(line[0], line[2]))
            bool_y_on_segment = (min(line[1], line[3]) <= project_point[1] <= max(line[1], line[3]))
            project_on_segment = bool_x_on_segment and bool_y_on_segment

            if not project_on_segment:
                continue
            poses_associtaion_counter[pose_idx] += 1
            mask[pose_idx, line_idx, n] = 1


            # one_to_n_matches.append([n, counter])
            # pose_indexes.append(pose_idx)
            # D2.append(project_point)
            # poses_points_correspondence.append([pose_idx, counter])

            # one_to_n_matches[counter,:] = np.asarray([n, counter])
            # pose_indexes[counter] = pose_idx
            # D2[counter,:] = project_point
            # poses_points_correspondence[counter, :] = np.asarray([pose_idx, counter])
            # counter += 1

    most_probable_pose_idx = np.argmax(poses_associtaion_counter) 
    max_num_associations = int(poses_associtaion_counter[most_probable_pose_idx])
    # print 'max_num_associations = {0}'.format(max_num_associations)

    if max_num_associations == 0:
        return one_to_n_matches, pose_indexes, D2, poses_points_correspondence, max_num_associations
 
    new_mask = mask[most_probable_pose_idx,:,:]  
    lines_points_idxs = np.where(new_mask == 1)
    # print lines_points_idxs
    for i in range(lines_points_idxs[0].shape[0]):
        line_idx = lines_points_idxs[0][i]
        point_idx = lines_points_idxs[1][i]
        project_point = np.asarray([projections[most_probable_pose_idx, line_idx, point_idx], \
                                    projections[most_probable_pose_idx, line_idx, point_idx + num_points]]) 
        # print 'one to n'
        # print np.asarray([point_idx, i])
        one_to_n_matches[i,:] = np.asarray([point_idx, i])
        pose_indexes[i] = most_probable_pose_idx
        D2[i,:] = project_point
        poses_points_correspondence[i, :] = np.asarray([most_probable_pose_idx, i])


    one_to_n_matches = one_to_n_matches[0:max_num_associations,:]
    pose_indexes = pose_indexes[0:max_num_associations] 
    D2 = D2[0:max_num_associations,:]
    poses_points_correspondence = poses_points_correspondence[0:max_num_associations,:]
    return one_to_n_matches, pose_indexes, D2, poses_points_correspondence, max_num_associations

def find_associations_from_map_before_icra(local_radar_points, local_radar_normals_radian, poses, intersect_segments, \
                               frame_idx, street_id, component_id, gt_pos):
    """
    # ICRA submission version 

    Imput:
        radar_points: 4 x N matrix [x, y, z, 1]^T, radar points
        poses: N x3 matrix [x, y, rotation]

    Output:
        D1:   3 x N matrix [x, y, z], local radar points
        D2 :  3 x M matrix, sample sample from map semantics
        associations: K x 2 matrix

    """
    start_time = time.time()
    visualization_offset = 0.2
    association_dist_thresh = MAX_PROJECT_DIST
    one_to_n_matches = []
    pose_indexes = []
    poses_points_correspondence = []
    poses_SE2 = np.zeros(( 3, 3, poses.shape[0]))
    for i in range(poses.shape[0]):
        poses_SE2[:,:,i] = np.asarray([[np.cos(poses[i,2]), -np.sin(poses[i,2]), poses[i,0]], \
                                       [np.sin(poses[i,2]),  np.cos(poses[i,2]), poses[i,1]], \
                                       [                 0,                   0,          1]])
        poses_points_correspondence.append([])

    project_mat = np.zeros((2, 3, intersect_segments.shape[0]))
    lines_normal_angle = np.zeros((intersect_segments.shape[0]))
    map_points_normal_angle = []
    for l in range(intersect_segments.shape[0]):
        line = intersect_segments[l,:]
        # Use y = mx + b to parameterize the line
        m = (line[3] - line[1]) / (line[2] - line[0])
        line_angle = np.arctan(m)
        lines_normal_angle[l] = wrap_angle(line_angle + np.pi/2)
        b = line[3] - m * line[2]
        project_mat[:,:,l] = np.asarray([[1,      m, -m * b],
                                         [m, m ** 2,      b]])
    num_intersect_segments = intersect_segments.shape[0]


    # print("\033[92m {}\033[00m".format(
    #         '###################################### Compute distance matrix ###################################'))
    # projections, distance_mat = projection_point_and_distances(local_radar_points, poses_SE2, \
    #                                                           project_mat, num_intersect_segments)

    # print 'done in {0}s.'.format(time.time() - start_time)
    # print("\033[92m {}\033[00m".format(
    #         '#########################################################################'))
    # Check based on normal
    counter = 0
    D1 = local_radar_points[0:3,:]
    D2 = []
    for n in range(local_radar_points.shape[1]):
        per_point_to_lines_dist = distance_mat[:,:,n]
        poses_lines_idxs = np.argwhere(per_point_to_lines_dist < association_dist_thresh)
        # print ''
        # print 'point {0} has {1} potential associations'.format(n, poses_lines_idxs.shape[0])
        for i in range(poses_lines_idxs.shape[0]):
            pose_idx = poses_lines_idxs[i,0]
            line_idx = poses_lines_idxs[i,1]
            project_point = projections[pose_idx, line_idx, :, n]
            line = intersect_segments[line_idx]
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            x3 = project_point[0]
            y3 = project_point[1]
            project_on_segment = (min(x1, x2) <= x3 <= max(x1, x2)) and (min(y1, y2) <= y3 <= max(y1, y2))
            if not project_on_segment:
                continue

            pose = poses[pose_idx, :]
            line_normal_global_angle = wrap_angle(lines_normal_angle[line_idx])
            point_normal_global_angle = wrap_angle(local_radar_normals_radian[n] + pose[2])
            normal_angle_diff = abs(line_normal_global_angle - point_normal_global_angle)
            # print 'Angle diff = {0}'.format(normal_angle_diff)
            if normal_angle_diff < NORMAL_DIFF_THRESH:
                # print '--------------------------------One match------------------------------'
                one_to_n_matches.append([n, counter])
                map_points_normal_angle.append(line_normal_global_angle)
                poses_points_correspondence[pose_idx].append(counter)
                D2.append(project_point)
                pose_indexes.append(pose_idx)

                counter += 1

    print 'Number of one_to_n_matches = {0}'.format(len(one_to_n_matches))
    if len(one_to_n_matches) > 0:
        D2 = np.asarray(D2)
        D2 = np.transpose(np.hstack(( D2 , np.ones((D2.shape[0], 1)))))
        associations = np.empty((0, 2))
        one_to_n_matches = np.asarray(one_to_n_matches)
        pose_indexes = np.asarray(pose_indexes)
        most_probable_pose_idx = np.bincount(pose_indexes).argmax()
        for map_point_idx in poses_points_correspondence[most_probable_pose_idx]:
            associations = np.vstack((associations , one_to_n_matches[map_point_idx,:]))
        print 'Number of initial associations = {0}'.format(associations.shape[0])
        D1_global = np.dot(poses_SE2[:, :, most_probable_pose_idx], local_radar_points[0:3,:])
        associations = associations.astype(int)
        top_global_pose = poses[most_probable_pose_idx,:]
    else:
        D1_global =[]
        D2 =[]
        associations = np.empty(0)
        top_global_pose = []
        most_probable_pose_idx = -1
    print 'find_associations_from_map done in {0}s.'.format(time.time() - start_time)

    return D1_global, D2, associations, top_global_pose, most_probable_pose_idx


def transition_new_street(pmu, pSigma, uk, Qk, vt_1, vt, mapgraph):
    vt_1_heading = mapgraph.get_road_heading(vt_1, [0.0])
    vt_heading = mapgraph.get_road_heading(vt, [0.0])
    se2_pmu_street_vt_1 = np.asarray([pmu[0,0], pmu[1,0], pmu[2,0]])
    uk_SE2 = se2_to_SE2(uk)
    SE2_pmu = se2_to_SE2(se2_pmu_street_vt_1)

    SE2_nmu_street_vt_1 = np.dot(SE2_pmu, uk_SE2)
    # print 'SE2_nmu_street_vt_1:'
    # print SE2_nmu_street_vt_1
    # print 'uk'
    # print uk
    # print 'uk_SE2'
    # print uk_SE2
    # print 'SE2_pmu'
    # print SE2_pmu
    if vt == vt_1:
        se2_nmu_street_vt_1 = SE2_to_se2(SE2_nmu_street_vt_1)
        nmu_vt = np.asarray([[se2_nmu_street_vt_1[0]],\
                             [se2_nmu_street_vt_1[1]],\
                             [se2_nmu_street_vt_1[2]]])
        nSigma = propagate_uncertainty(pmu, uk, pSigma, Qk)

    else:
        vt_origin = mapgraph.node[vt]['origin']
        vt_1_origin = mapgraph.node[vt_1]['origin']
        SE2_vt = se2_to_SE2(np.asarray([vt_origin[0], vt_origin[1], vt_heading]))
        SE2_vt_1 = se2_to_SE2(np.asarray([vt_1_origin[0], vt_1_origin[1], vt_1_heading]))
        SE2_from_vt_to_vt_1 = np.dot(inverse_pose(SE2_vt),SE2_vt_1)
        SE2_nmu_vt = np.dot(SE2_from_vt_to_vt_1, SE2_nmu_street_vt_1)
        se2_nmu_street_vt = SE2_to_se2(SE2_nmu_vt)

        nmu_vt = np.asarray([[se2_nmu_street_vt[0]],\
                             [se2_nmu_street_vt[1]],\
                             [se2_nmu_street_vt[2]]])

        moveA = np.eye(3)
        moveA[0:2,0:2] = SE2_from_vt_to_vt_1[0:2,0:2]
        pSigma = np.dot(moveA, np.dot(pSigma, moveA.T))
        nSigma = propagate_uncertainty(pmu, uk, pSigma, Qk)




    return nmu_vt, nSigma

def transition_old_street(pmu, pmuY, odom, odom_y):
    if abs(pmu[2, 0]) < (np.pi / 2):
        A_odom = 1.0
    else:
        A_odom = -1.0
    A_s = np.array([[1, 0, 0, 0], \
                    [1, 0, 0, 0], \
                    [0, 0, 1, 0], \
                    [0, 0, 1, 0]])
    b_s = np.array([[A_odom*odom[0]],\
                               [0.0],\
                           [odom[1]],\
                              [0.0]])
    nmu = np.dot(A_s, pmu) + b_s
    nmuOffset = pmuY + odom_y
    return nmu, nmuOffset

def longitudinal_correction(normal_radians):
    """Check if there are lateral information captured by the radar points so that we can correct the drift
    on longitudinal direction, i.e., if the biggest eigen vector is too dominant.
    """
    # covariance = np.cov(points[0:2,:])
    # w, v = np.linalg.eig(covariance)
    # min_eig_idx = np.argmin(w)
    # max_eig_idx = np.argmax(w)
    # eigen_value_dominance = np.array([w[max_eig_idx] / w[min_eig_idx]])
    # print ('eigen_value_dominance: ' + str(eigen_value_dominance))
    # if eigen_value_dominance > 3.0:
    #     return False
    # else:
    #     return True
    normal_angles = abs(normal_radians) * 180 / np.pi
    angle_threshold = 20 # degree
    num_longitudinal_correction = normal_angles[np.where( normal_angles < (90 - angle_threshold) ) ].shape[0] + \
                                  normal_angles[np.where( normal_angles > (90 + angle_threshold) ) ].shape[0]
    percentage = num_longitudinal_correction / float(normal_angles.shape[0])
    if debugFilter:
        print ('longitudinal correction percentage: ' + str(percentage) +\
               ' with threshold = ' + str(longitudinal_num_correction_thresh))
    if percentage > longitudinal_num_correction_thresh:
        print 'Apply Longitudinal Correction'

        return True
    else:
        return False




# @jit(nopython=True, target='cuda')
# @jit(nopython=True)
@njit('float64(float64[:], float64[:],float64[:])')
def check_p2l_intersection_cpu(line_origin, line_terminus, point):
    """Check whether point projects onto line """
    b_intersect = True
    # distance between p1 and p2
    # l2 = (line_origin[0] - line_terminus[0])**2 + (line_origin[1] - line_terminus[1])**2
    l2 = np.sum((line_origin-line_terminus)**2)
    # if you need the point to project on line extention connecting p1 and p2
    temp_a = point - line_origin
    temp_b = line_terminus - line_origin
    temp_c = temp_a * temp_b
    t = np.sum(temp_c) / l2
    if t > 1 or t < 0:
        # p3 does not project onto p1-p2 line segment
        b_intersect = False

    return b_intersect


# @jit(nopython=True, target='cuda')
# @guvectorize(['void(float64[:,:], int32[:],float64[:,:], float64)'], \
# '(m,n),(k),(r,c),() -> ()',target ="cuda")
# @jit(nopython=True, target='cuda')
# @cuda.jit
@njit('float64(float64[:,:], int64[:], float64[:,:])')
def points_weighting_cpu1(points, intersect_segments_idx, all_segments):
    distSum = np.float64(0.0)
    weight = np.float64(0.0)
    i_n_loop = points.shape[1]
    j_n_loop = intersect_segments_idx.shape[0]

    for i in range(i_n_loop):
        p = points[:,i]
        for j in range(j_n_loop):
            line = all_segments[intersect_segments_idx[j]]
            line_start = line[0:2]
            line_end = line[2:4]
            b_intersect = check_p2l_intersection_cpu(line_start, line_end, p)
            if  b_intersect:
                dist = projection_distance(line_start, line_end, p)
                if dist < MAX_PROJECT_DIST:
                    distSum += dist # The larger the distance, the smaller the weight
    weight = 1.0 / distSum
    return weight

def points_weighting_cpu2(option, points, points_rtree, normals_radians, intersect_segments_idx, all_segments):
    intersection_offset = 0.01 # (km)
    
    points_weights = np.zeros(points.shape[1], dtype=np.float64)
    for i in range(intersect_segments_idx.shape[0]):
        line = all_segments[intersect_segments_idx[i]]
        if line[0] < line[2]:
            x_min = line[0]
            x_max = line[2]
        else:
            x_min = line[2]
            x_max = line[0]
        if line[1] < line[3]:
            y_min = line[1]
            y_max = line[3]
        else:
            y_min = line[3]
            y_max = line[1]

        # query_time = time.time()
        points_in_box_idx = np.array(list(points_rtree.intersection((x_min-intersection_offset,\
                                                                y_min-intersection_offset,\
                                                                x_max+intersection_offset,\
                                                                y_max+intersection_offset))),
                                                                dtype=np.int64)
        # print 'Rtree query done in {0}s.'.format(time.time() - query_time)
        points_weights_temp = compute_weight_per_line(points, normals_radians,\
                                                      points_in_box_idx,line)
        points_weights = points_weights + points_weights_temp
        # print points_weights_temp
    return points_weights

@njit
def uniform_likelihood(distance, offset):
    likelihood = np.float64(0.0)
    likelihood = 1.0 / offset
    return likelihood

@njit
def gaussian_likelihood(distance, std):
    # The larger the distance, the smaller the weight
    if distance > std:
        likelihood = 0.0
        return likelihood
    likelihood = np.float64(0.0)
    likelihood = np.exp(-0.5*(distance/std)**2)
    return likelihood


@njit('float64[:](float64[:,:], float64[:],int64[:], float64[:])')
def compute_weight_per_line(points, normals_radians, points_in_box_idx, line):
    # print 'Number of points in line: {0}'.format(points_in_box_idx.shape[0])

    line_start = line[0:2]
    line_end = line[2:4]
    likelihoods = np.zeros(points.shape[1], dtype=np.float64)
    l = line_end - line_start
    line_normal = np.arctan2(l[1], l[0]) + np.pi / 2.0
    # print 'line normal {0}'.format(line_normal)
    if line_normal < 0:
        line_normal = line_normal + np.pi
    for i in range(points_in_box_idx.shape[0]):
        # Check if the point project onto the line
        idx = points_in_box_idx[i]
        pt = points[:,idx]
        point_normal = normals_radians[idx]

        if point_normal < 0:
            point_normal = point_normal + np.pi

        b_intersect = check_p2l_intersection_cpu(line_start, line_end, pt)
        if b_intersect:
            dist = projection_distance(line_start, line_end, pt)
            normals_diff = abs(line_normal - point_normal)
            # check 1. the point to line distance 2. if the point normal similar to the line normal
            if dist < MAX_PROJECT_DIST and  normals_diff < NORMAL_DIFF_THRESH:
                # _likelihood = uniform_likelihood(dist, 1.0)
                _likelihood = gaussian_likelihood(dist, MAX_PROJECT_DIST)

                if point_normal < ((30/180)*np.pi) or point_normal > ((150/180)*np.pi):
                    _likelihood = _likelihood * 100.0 # to promote point with correction information
                    # print ('likelihood: '),
                    # print (_likelihood)
                likelihoods[idx] = _likelihood

    return likelihoods

def draw_pose(options, frame_idx, compId, mapgraph, street, gt_pos, theta, weight, intersect_segments_idx, all_segments):
    # type: (object, object, object, object, object, object, object, object, object, object) -> object
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    lines = []

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    street_node =  mapgraph.node[street]
    intersection_offset = 0.2  # 0.2 / project_scale * 1000 (m)

    ax.plot([street_node['origin'][0], street_node['terminus'][0]],\
            [street_node['origin'][1], street_node['terminus'][1]],\
            linewidth=0.3, c='black')
    if gt_pos is not None:
        ax.scatter(gt_pos[0], gt_pos[1], s=0.2, c='r', label='gt')
        ax.set_xlim([gt_pos[0] - intersection_offset, gt_pos[0] + intersection_offset])
        ax.set_ylim([gt_pos[1] - intersection_offset, gt_pos[1] + intersection_offset])
    curr_pos = trans_pose_global( mapgraph, street, theta)[0]
    ax.scatter(curr_pos[0], curr_pos[1], s=0.2, c='b', label='mu')

    for i in range(intersect_segments_idx.shape[0]):
        line = all_segments[intersect_segments_idx[i]]
        lines.append(np.vstack((line[0:2], line[2:4])))

    ax.add_collection(
            LineCollection(lines, transOffset=ax.transData, linewidths=0.1, colors='green'))

    print 'weight = {0}'.format(weight)
    ax.set_title('Weight: ' + str(weight) + '\nStreet: ' + street)
    ax.legend()
    directory = cwd + '/results/initial_result/' + str(frame_idx).zfill(6) + '_'
    fig_name = directory + str(compId) + '_' + '_onePost.png'
    fig.savefig(fig_name, dpi=400)

def draw_points_lines(ax,street, gt_pos, curr_pos, curr_heading, weight, max_weight, max_number, \
                      points_global, points_weight, normals_local, intersect_segments_idx, all_segments):

    ax.plot([street['origin'][0], street['terminus'][0]], [street['origin'][1], street['terminus'][1]],\
            linewidth=0.3, c='black')
    lines = []

    if gt_pos is not None:
        ax.scatter(gt_pos[0], gt_pos[1], s=0.2, c='r', label='gt')
    # plt.plot(street['origin'][0], street['origin'][1], '^')
    # plt.plot(street['terminus'][0], street['terminus'][1], 'x')
    ax.scatter(curr_pos[0], curr_pos[1], s=0.2, c='b', label='mu')
    ax.quiver(curr_pos[0], curr_pos[1], np.cos(curr_heading), np.sin(curr_heading), width=0.002,color='b')
    survived_idx = np.where(points_weight > 0)[0]
    normals_survived = normals_local[:,survived_idx]
    points_survived = points_global[:,survived_idx]
    normals_global = np.array([[np.cos(curr_heading), -np.sin(curr_heading)],
                               [np.sin(curr_heading),  np.cos(curr_heading)]]).dot(normals_local[0:2,:])
    ax.quiver(points_global[0, :], points_global[1,:],\
              normals_global[0, :], normals_global[1, :], width=0.002, alpha=0.2, color='black')
    normals_global_survived = np.array([[np.cos(curr_heading), -np.sin(curr_heading)],
                               [np.sin(curr_heading),  np.cos(curr_heading)]]).dot(normals_survived[0:2,:])
    
    ax.quiver(points_survived[0, :], points_survived[1,:],\
              normals_global_survived[0, :], normals_global_survived[1, :], width=0.002)
    ax.scatter(points_global[0, :], points_global[1, :], s=1, c = points_weight, edgecolors='None')

    for i in range(intersect_segments_idx.shape[0]):
        line = all_segments[intersect_segments_idx[i]]
        lines.append(np.vstack((line[0:2], line[2:4])))

    ax.add_collection(
            LineCollection(lines, transOffset=ax.transData, linewidths=0.1, colors='green'))
    ax.set_title('Weight: ' + str(weight)[0:7] + ' with max weight: ' + str(max_weight)[0:7] + \
                    '\nNumber of matched points: ' + str(points_survived.shape[1]) + \
                    ' with max number:' + str(max_number))
    ax.legend()
    # directory = cwd + '/results/initial_result/oneSample_' + str(frame_idx).zfill(6)
    #
    # fig_name = directory + '_' + str(sample_id).zfill(3) + '.png'
    # fig.savefig(fig_name, dpi=400)
    return ax




class RadarFilter:
    def __init__(self):
        # For visualization
        self.weight_list = []
        self.points_global_list = []
        self.all_points_weights = []
        self.pos_list = []
        self.heading_list = []
        self.intersect_segment_list = []
        self.streets = []
        self.counter = 0
        self.last_uk = []
        self.initialized = False

    def updateSampling(self, options, mapgraph, street, yt, obsA, obsb, obsOdomSigma, \
                       nmu, nSigma, nmuOffset, nSigmaOffset,\
                    b_longitudinal_correction, local_radar_points, local_points_rtree, local_radar_normals_radian, \
                       all_segments, intersect_segments):
        """
            Update method using sampling: sample from the know prior (predicted) gaussian distribution,
            use the radar points to compute the mean and the sigma of the observation gaussian distribution
        """

        sample_number = 10
        street_node = mapgraph.node[street]
        thetas_offsets = self.gaussian_sampling(b_longitudinal_correction, nmu, nmuOffset, nSigma, nSigmaOffset, \
                                                    sample_number)

        # Todo: implement importance sampling here since we sample from gaussian
        weights = np.zeros(sample_number)
        poses = trans_pose_global( mapgraph, street, thetas_offsets)
        for j in range(thetas_offsets.shape[0]):
            global_position = poses[j, 0:2]
            global_heading = poses[j, 2]
            weight = self.evaluate_one_sample(options, mapgraph, street_node, \
                                 global_heading, global_position, \
                                 local_radar_points, local_points_rtree, local_radar_normals_radian,\
                                 all_segments, intersect_segments)
            weights[j] = weight
        weight_sum = np.sum(weights)
        weights = weights / weight_sum

        # Compute offset observation distribution
        muOffsetHat = np.zeros((1,1))
        SigmaOffsetHat = np.zeros((1,1))
        for j in range(thetas_offsets.shape[0]):
            muOffsetHat = weights[j] * thetas_offsets[j, 4] + muOffsetHat
        for j in range(thetas_offsets.shape[0]):
            SigmaOffsetHat = weights[j] * (thetas_offsets[j, 4] - muOffsetHat)**2 + SigmaOffsetHat


        if b_longitudinal_correction:
            # Compute theta observation distribution, i.e., distance, previous distance, angle, previous angle
            print ('Apply longitudinal correction')

            obst = np.zeros((2,1)) # observation: [relative distance, relative angle] 2 x 1 matrix
            varY = np.zeros(2) #
            obsRadarSigma = np.zeros((2,2))
            for j in range(thetas_offsets.shape[0]):
                obst[0][0] = weights[j] * (thetas_offsets[j, 0] - thetas_offsets[j, 1]) + obst[0]
                obst[1][0] = weights[j] * (thetas_offsets[j, 2] - thetas_offsets[j, 3]) + obst[1]
            for j in range(thetas_offsets.shape[0]):
                varY[0] = weights[j] * (thetas_offsets[j, 0] - thetas_offsets[j, 1] - obst[0]) ** 2 + varY[0]
                varY[1] = weights[j] * (thetas_offsets[j, 2] - thetas_offsets[j, 3] - obst[1]) ** 2 + varY[1]
            obsRadarSigma[0,0] = varY[0]
            obsRadarSigma[1,1] = varY[1]
            print 'obsRadarSigma = {0}'.format(obsRadarSigma)
            (c, muHat, SigmaHat) = gaussian_dist_yxxmu_product_x(obst, obsA, obsb, obsRadarSigma, \
                                                             nmu, nSigma)
            logC = c
        else:
            obst = yt
            (logC, muHat, SigmaHat) = gaussian_dist_yxxmu_product_x(obst, obsA, obsb, obsOdomSigma, \
                                                                 nmu, nSigma)

        return (logC, muHat, SigmaHat, muOffsetHat, SigmaOffsetHat)


    def updateScanMatching(self, options, mapgraph, street, nmu, nSigma, nmuOffset, nSigmaOffset, \
                           local_radar_points, local_points_rtree, all_segments):
        """update method 2: perform scan to map matching (maximize a prior)"""
        logC = 0.0
        muHat = 0.0
        SigmaHat = np.identity(nSigma.shape[0])
        return (logC, muHat, SigmaHat)

    def scan_matching(self):
        pose = []
        return pose


    def global_observation(self, options, frame_idx, mapgraph, mapdynamics, \
                             post_previous, allStreets, odom_x_angle, odom_y, local_radar_points, \
                             local_radar_normals_radian, all_segments, gt_pos):

        uk_from_t_1_to_t_se2 = np.asarray([odom_x_angle[0,0], odom_y, odom_x_angle[1,0]])
        Qk = np.asarray([[moveVarX,          0.0,            0.0],\
                         [     0.0,     moveVarY,            0.0],\
                         [     0.0,          0.0,   moveVarAngle]])

        # Parameters
        topKstreets = 2
        sample_number = 100
        intersection_offset = 0.15  # km
        small_prob_street_thres = 0.01

        if 'oxford' in options.dataname:
            LANE_ERROR = LANE_ERROR_OXFORD

        elif 'mulran' in options.dataname:
            LANE_ERROR = LANE_ERROR_MULRAN

        elif 'boreas' in options.dataname:
            LANE_ERROR = LANE_ERROR_BOREAS

        else:
            LANE_ERROR = 0.5 / 1000

        best_global_pose = np.zeros((1, 3))
        best_street = []
        best_prob = 0.0
        b_longitudinal_correction = False
        b_lateral_correction = False
        max_num_associations = 0

        global_pose_sigma = np.eye(3) * 0.01
        streetWeights = np.zeros(len(allStreets))
        streets = []

        for count, street in enumerate(allStreets):
            streets.append(street)

            if street == SINK_ROAD:
                continue
            if not np.isfinite(post_previous.logV[street]):
                continue

            streetWeights[count] = np.exp(post_previous.logV[street])
        indices = (-streetWeights).argsort()[:topKstreets]

        # Predict probability to new streets
        street_probs = []
        new_predicted_streets = []
        nmu_list = []
        nSigma_list = []

        for street_idx in indices[0:min(topKstreets, len(streets))]:
            vt_1 = streets[street_idx]
            prevThetaDist = post_previous.Theta[vt_1]
            if vt_1 == SINK_ROAD:
                continue
            if not np.isfinite(post_previous.logV[vt_1]):
                continue
            pmu = prevThetaDist.getMu(0)
            pSigma = prevThetaDist.getSigma(0)
            vtIndices = mapgraph.node[vt_1]['transition_node_index']
            for (vt, transI) in vtIndices.iteritems():
                if vt == SINK_ROAD:
                    continue
                if 'origin' not in mapgraph.node[vt]:
                    continue
                nmu, nSigma = transition_new_street(pmu, pSigma, uk_from_t_1_to_t_se2, Qk, \
                                                    vt_1, vt, mapgraph)
                vt_length = mapgraph.node[vt]['length']
                # determinie if we need to skip this street

                origin_heading = mapgraph.get_road_heading(vt, [0.0])[0]
                terminus_heading = mapgraph.get_road_heading(vt, [vt_length])[0]
                if origin_heading == terminus_heading:
                    # line
                    if heading_thres < abs(nmu[2, 0]) < (2 * np.pi - heading_thres):
                        # print 'angle = {0}, skip this line transition street for observation sampling'.format(nmu[2,0])
                        # print heading_thres
                        continue
                else:
                    # curve
                    if heading_thres < abs(nmu[2, 0]) < (2 * np.pi - heading_thres) \
                            and heading_thres < abs(nmu[2, 0] + origin_heading - terminus_heading) < (
                            2 * np.pi - heading_thres):
                        # print 'angle = {0}, skip this curve street for observation sampling'.format(nmu[2,0])
                        continue


                # Get lane information
                min_lateral_y, max_lateral_y = self.sampling_lateral_range(options, vt, mapgraph, all_segments, LANE_ERROR)

                # Restrict samples based on lane information
                low_x = np.array([0.0, max_lateral_y])
                low_y = np.array([vt_length, min_lateral_y])
                upp = np.array([vt_length, max_lateral_y])
                mu = np.array([ nmu[0,0], nmu[1,0]])
                cov = nSigma[0:2,0:2]
                cdf_value_low_x = stats.multivariate_normal.cdf(low_x, mu, cov)
                cdf_value_low_y = stats.multivariate_normal.cdf(low_y, mu, cov)
                cdf_value_upp = stats.multivariate_normal.cdf(upp, mu, cov)
                # print 'mu = {0}, cov = {1}, low = {2}, up = {3}'.format(mu, cov, low_x, upp)
                # print 'street = {0}'.format(vt),
                # print ' cdf_low_x = {0}, cdf_low_y = {1}, cdf_upp = {2}'.format(cdf_value_low_x, cdf_value_low_y, cdf_value_upp)
                # print ''
                prob_ = cdf_value_upp - cdf_value_low_x - cdf_value_low_y
                new_predicted_streets.append(vt)
                street_probs.append(prob_)
                nmu_list.append(nmu)
                nSigma_list.append(nSigma)
        # Top K streets to sample from
        indices = (-np.array(street_probs)).argsort()[:topKstreets]
        # print 'street_probs = {0}'.format(street_probs)
        # print 'sorted indices = {0}'.format(indices)
        flag_high_prob_street = False
        for idx in indices:
            if street_probs[idx] > small_prob_street_thres:
                flag_high_prob_street = True



        # Todo ------------------------------------------------------------------------------

        # Todo ------------------------------------------------------------------------------
        # import matplotlib
        # matplotlib.use('agg')
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # lg_colors = ['green', 'yellow', 'purple', 'blue']
        if len(indices) > 0:
            highest_street = new_predicted_streets[indices[0]]
        for count, idx in enumerate(indices):

            i = 0
            nmu = nmu_list[idx]
            nSigma = nSigma_list[idx]
            vt = new_predicted_streets[idx]
            prob_ = street_probs[idx]
            if prob_ < small_prob_street_thres and flag_high_prob_street:
                continue

            # is the same street to the highest prob street, skip
            if count > 0 and vt == highest_street:
                continue

            curr_pos = trans_pose_global(mapgraph, vt, nmu.reshape((1, 3)))[0]

            intersect_segments_idx = np.asarray(list(\
                mapgraph.rtreeIndex.intersection((curr_pos[0] - intersection_offset, \
                                                  curr_pos[1] - intersection_offset, \
                                                  curr_pos[0] + intersection_offset, \
                                                  curr_pos[1] + intersection_offset))))
            # ax.plot([mapgraph.node[vt]['origin'][0], mapgraph.node[vt]['terminus'][0]], \
            #         [mapgraph.node[vt]['origin'][1], mapgraph.node[vt]['terminus'][1]], \
            #         linewidth=0.3, c=lg_colors[count])
            # ax.scatter(curr_pos[0], curr_pos[1], s=0.2, c=lg_colors[count])

            # Sample local thetas
            thetas = self.gaussian_sampling(True, nmu, nSigma, sample_number)
            vt_1 = vt
            # filter some samples 
            min_lateral_y, max_lateral_y = self.sampling_lateral_range(options, vt, mapgraph, all_segments, LANE_ERROR)
            if 'lanes' in mapgraph.node[street]:
                num_lanes = float(mapgraph.node[vt]['lanes'])
            else:
                num_lanes = 1.0
            print 'Number of lanes = {0}'.format(num_lanes)
 
            thetas_y = thetas[:, 1]
            keep_idxes = np.argwhere((thetas_y < max_lateral_y) & (thetas_y > min_lateral_y))
            if keep_idxes.shape[0] == 0:
                # This is worst case. We try to recover from the lane information
                print 'No observation samples from prior dist of street, draw samples from street distribution'
                y_samples = np.random.uniform(min_lateral_y, max_lateral_y, sample_number)
                thetas[:, 1] = y_samples
            else:
                keep_idxes = keep_idxes.reshape(keep_idxes.shape[0])
                thetas = thetas[keep_idxes, :]
            poses = trans_pose_global(mapgraph, vt_1, thetas)
            intersect_segments = all_segments[intersect_segments_idx]
            before_filter_num = intersect_segments.shape[0]

            # Remove some segments base on distance and projection
            if before_filter_num > 600:
                street_origin = mapgraph.node[vt]['origin']
                street_terminus = mapgraph.node[vt]['terminus']
                street_angle = np.arctan2(street_terminus[1] - street_origin[1], \
                                          street_terminus[0] - street_origin[0])
                seg_start_time = time.time()
                intersect_segments = filter_segments(intersect_segments, curr_pos, \
                    street_angle, intersection_offset)
                # print 'filter_segments done in {0}s. Before: {1}'.format((time.time() - seg_start_time), before_filter_num)


            # Find associations respect to the OSM map
            radar_points_global, map_points_global, \
            associations, top_global_pose, top_pose_idx = find_associations_from_map(local_radar_points, \
                                                                local_radar_normals_radian, poses, intersect_segments, \
                                                                frame_idx, street_idx, i, gt_pos)
            # Evaluate the associations
            b_long_temp, b_lateral_temp, \
            long_features_idx, lateral_features_idx = evaluate_associations(local_radar_points, \
                                                                            local_radar_normals_radian, radar_points_global, map_points_global, associations)
            # if b_long_temp and b_lateral_temp:
            num_associations = associations.shape[0]
            print 'num_associations = {0}, prob_ = {1}'.format(num_associations, prob_)
            if num_associations <= 0:
                continue

            if num_associations > max_num_associations:
                max_num_associations = associations.shape[0]
                b_longitudinal_correction = b_long_temp
                b_lateral_correction = b_lateral_temp
                best_radar_points_global = radar_points_global
                best_map_points_global = map_points_global
                best_associations = associations
                best_global_pose = top_global_pose
                best_street_id = street_idx
                best_component_id = i
                best_long_features_idx = long_features_idx
                best_lateral_features_idx = lateral_features_idx
                best_street = vt_1
                # best_prob = prob_
                best_theta = thetas[top_pose_idx,:]


        visualization_offset = 0.2
        # ax.scatter(gt_pos[0], gt_pos[1], s=0.2, c='red')
        # ax.legend()
        # ax.set_xlim([gt_pos[0] - visualization_offset, gt_pos[0] + visualization_offset])
        # ax.set_ylim([gt_pos[1] - visualization_offset, gt_pos[1] + visualization_offset])
        # directory = cwd + '/results/visualization/' + options.dataname
        # fig_name = directory + '/observation_streets_' + str(frame_idx).zfill(4) + '_'  '.png'
        # fig.savefig(fig_name, dpi=200)
        # plt.close()
        if max_num_associations > 0:
            if drawObservation:
                debug_visualize_global_observation(options, best_radar_points_global, best_map_points_global, local_radar_normals_radian, \
                                           best_global_pose, intersect_segments, best_associations, \
                                           gt_pos, frame_idx, best_street_id, best_component_id, \
                                           b_longitudinal_correction, b_lateral_correction,\
                                           np.asarray(best_long_features_idx), np.asarray(best_lateral_features_idx),\
                                               mapgraph, best_street)
            # print 'best theta = {0}, best street = {1}, max_num_associations = {2} '.format(best_theta, best_street,\
            #                                                                                 max_num_associations)
        else:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            directory = cwd + '/results/visualization/' + options.dataname + \
                '/debug_point_line_normal_' + str(frame_idx).zfill(4) 
            fig_name = directory +'.png'
            fig.savefig(fig_name, dpi=displayDPI)
            plt.close()
            best_global_pose = np.zeros((1,3))
            b_longitudinal_correction = False
            b_lateral_correction = False
        # print 'b_longitudinal_correction = {0}, b_lateral_correction = {1}'.format(b_longitudinal_correction, \
        #                                    b_lateral_correction)
        return best_global_pose, global_pose_sigma, b_longitudinal_correction, b_lateral_correction

    def sampling_lateral_range(self, options, street, mapgraph, all_segments, LANE_ERROR):
        """ Restrict samples based on lane and surrounding structures information """
        street_length = mapgraph.node[street]['length']
        # Get lane information
        if 'lanes' in mapgraph.node[street]:
            num_lanes = float(mapgraph.node[street]['lanes'])
        else:
            num_lanes = 1.0

        # print 'Number of lanes = {0}'.format(num_lanes)
        if debugFilter:
            print 'observation street is = {0}'.format(street)
        if num_lanes == 1.0:

            if options.drive_side == 'drive_left':
                y_max =  num_lanes * LANE_WIDTH
                y_min = -LANE_ERROR              
            else:
                y_max =  num_lanes * LANE_WIDTH / 2.0 + LANE_ERROR
                y_min = -num_lanes * LANE_WIDTH / 2.0 - LANE_ERROR
        elif num_lanes == 4.0:
            if options.drive_side == 'drive_left':
                y_max =  num_lanes * LANE_WIDTH / 2.0
                y_min = -num_lanes * LANE_WIDTH / 2.0                
            else:
                y_max =  num_lanes * LANE_WIDTH / 2.0 + LANE_ERROR
                y_min = -num_lanes * LANE_WIDTH / 2.0 - LANE_ERROR            
        elif num_lanes == 3.0:
            # num lanes 3
            if options.drive_side == 'drive_left':
                y_max =  1.5 * LANE_WIDTH
                y_min = -1.5 * LANE_WIDTH
            else:
                y_max =  1.5 * LANE_WIDTH
                y_min = -1.5 * LANE_WIDTH
        else:
            # num lanes 2
            if options.drive_side == 'drive_left':
                y_max =  num_lanes * LANE_WIDTH
                y_min = -LANE_ERROR
            else:
                y_max = LANE_ERROR 
                y_min = -num_lanes * LANE_WIDTH

        return  y_min, y_max

    def evaluate_one_sample(self, options, mapgraph, street_node, global_heading, global_position, \
                            local_radar_points, local_points_rtree, local_radar_normals_radians, \
                            all_segments, intersect_segments):
        points_global = trans_points_global(local_radar_points, global_heading, global_position)
        all_segments_local = trans_lines_local(all_segments, global_heading, global_position)

        points_weights = points_weighting_cpu2(options, local_radar_points[0:2, :], \
                                               local_points_rtree, local_radar_normals_radians, \
                                               intersect_segments, all_segments_local)
        weight = np.sum(points_weights)
        # print 'weight = {0}'.format(weight)
        # print 'Weighting one sample done in {0}s with weight: {1}.\n'.format((time.time() - weight_time), weight)
        self.weight_list.append(weight)
        self.pos_list.append(global_position)
        self.heading_list.append(global_heading)
        self.intersect_segment_list.append(intersect_segments)
        self.points_global_list.append(points_global)
        self.all_points_weights.append(points_weights)
        self.streets.append(street_node)
        return weight


    def uniform_sampling(self, mu, offset, sigma_theta, sigma_offset, sample_number):
        """
        Sample from known distribution
        Output:
            samples:
            [distance_now, distance_previous, angle_now, angle_previous, offset] N x 5 matrix
        """
        samples = np.zeros((sample_number, 5))
        mu = mu.reshape(4)
        var_0 = np.sqrt(sigma_theta[0, 0])
        var_1 = np.sqrt(sigma_theta[1, 1])
        var_2 = np.sqrt(sigma_theta[2, 2])
        var_3 = np.sqrt(sigma_theta[3, 3])
        var_4 = np.sqrt(sigma_offset[0, 0])

        samples[:, 0] = np.random.uniform(mu[0] - var_0, mu[0] + var_0, sample_number)
        samples[:, 1] = np.random.uniform(mu[1] - var_1, mu[1] + var_1, sample_number)
        samples[:, 2] = np.random.uniform(mu[2] - var_2, mu[2] + var_2, sample_number)
        samples[:, 3] = np.random.uniform(mu[3] - var_3, mu[3] + var_3, sample_number)
        samples[:, 4] = np.random.uniform(offset[0, 0] - var_4, offset[0, 0] + var_4, sample_number)

        return samples
    def gaussian_sampling(self, b_longitudinal_correction, mu, sigma, sample_number):
        """
        Sample from known gaussian distribution
        Input:
            mu: [curr_x, curr_y, curr_angle]

        Output:
            samples:
            [x, y, angle] N x 3 matrix
        """
        # print mu
        mu = mu.reshape(3)
        curr_prev_samples = np.random.multivariate_normal(mu, sigma, sample_number)
        samples = np.zeros((sample_number, 3))
        if b_longitudinal_correction:
            samples[:,0] = curr_prev_samples[:,0]
            samples[:,1] = curr_prev_samples[:,1]
            samples[:,2] = curr_prev_samples[:,2]

        else:
            samples[:, 0] = mu[0]
            samples[:,1] = curr_prev_samples[:,1]
            samples[:,2] = curr_prev_samples[:,2]

        return samples

    def gaussian_sampling_old(self, b_longitudinal_correction, mu, offset, sigma_theta, sigma_offset, sample_number):
        """
        Sample from known gaussian distribution

        Output:
            samples:
            [distance_now, distance_previous, angle_now, angle_previous, offset] N x 5 matrix
        """
        if b_longitudinal_correction:
            sigma = np.zeros((5, 5))
            mu = mu.reshape(4)
            mean = np.zeros(5)
            mean[0:4] = mu
            mean[4] = offset
            sigma[0:4, 0:4] = sigma_theta
            sigma[4, 4] = sigma_offset
            samples = np.random.multivariate_normal(mean, sigma, sample_number)
        else:
            sigma = np.zeros((3, 3))
            mu = mu.reshape(4)
            samples = np.ones((sample_number, 5))
            samples[:, 0] = mu[0]
            samples[:, 1] = mu[1]
            mean = np.zeros(3)
            mean[0:2] = mu[2:4]
            mean[2] = offset
            sigma[0:2, 0:2] = sigma_theta[2:4, 2:4]
            sigma[2, 2] = sigma_offset
            angle_offset_samples = np.random.multivariate_normal(mean, sigma, sample_number)
            samples[:, 2:5] = angle_offset_samples
        return samples

    def lateral_sampling(self, mu, sample_number):
        """
        We don't know anything about the lateral distribution,
        try to establish the gaussian for the first filtering iteration

        Input:
            mu: mean theta, i.e., [distance_now, distance_previous, angle_now, angle_previous]
        Output:
            samples:
            [distance_now, distance_previous, angle_now, angle_previous, offset] N x 5 matrix
        """
        small_offset = 0.0001

        samples = np.zeros((sample_number, 5))
        distance_now_samples = np.ones(sample_number) * mu[0]
        distance_previous_samples = np.ones(sample_number) * mu[1]
        heading_now_samples = np.ones(sample_number) * mu[2]
        heading_previous_samples = np.ones(sample_number) * mu[3]
        offset_samples = np.linspace(small_offset, ONE_WAY_WIDTH, sample_number).reshape(sample_number)

        samples[:, 0] = distance_now_samples
        samples[:, 1] = distance_previous_samples
        samples[:, 2] = heading_now_samples
        samples[:, 3] = heading_previous_samples
        samples[:, 4] = offset_samples
        return samples

    def draw_sample(self, frame_idx, gt_pos, local_normals, all_segments):
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        weight_sum = 0.0
        # Todo: Visualize some samples
        if debugFilter and (frame_idx) >= 100:
            max_weight = 0.0
            max_number_of_points = 0
            max_w_idx = -1
            max_n_idx = -1
            for idx, w in enumerate(self.weight_list):
                weight_sum += w
                if w > max_weight:
                    max_weight = w
                    max_w_idx = idx
                points_weight = self.all_points_weights[idx]
                survived_idx = np.where(points_weight > 0)[0]
                number_of_points = survived_idx.shape[0]
                if number_of_points > max_number_of_points:
                    max_number_of_points = number_of_points
                    max_n_idx = idx
            max_weight = max_weight
            indexes = [max_w_idx]
            for idx in indexes:
                fig = plt.figure(figsize=displayFigSz)
                ax = fig.add_subplot(111)
                intersect_segments = self.intersect_segment_list[idx]
                global_position = self.pos_list[idx]
                heading = self.heading_list[idx]
                weight = self.weight_list[idx]
                points_global = self.points_global_list[idx]
                points_weights = self.all_points_weights[idx]
                street_curr = self.streets[idx]
                ax = draw_points_lines(ax, street_curr, gt_pos, global_position, heading, weight,
                                       max_weight, max_number_of_points, points_global, points_weights, \
                                       local_normals, intersect_segments, all_segments)
                ax.set_xlim([gt_pos[0] - 0.2, gt_pos[0] + 0.2])
                ax.set_ylim([gt_pos[1] - 0.2, gt_pos[1] + 0.2])
                directory = cwd + '/results/initial_result/samples_' + str(frame_idx).zfill(6)
                fig_name = directory + '_' + str(idx) + '.png'
                fig.savefig(fig_name, dpi=100)
                plt.close('all')

        self.weight_list = []
        self.points_global_list = []
        self.all_points_weights = []
        self.pos_list = []
        self.heading_list = []
        self.intersect_segment_list = []
        self.streets = []
        self.counter += 1
    
    
    
    def updateStep_old(self, b_longitudinal_correction, b_lateral_correction, local_x_y_angle, nmu, nSigma):
        nnmu = np.copy(nmu)
        nnSigma = np.copy(nSigma)

        if b_longitudinal_correction and b_lateral_correction:
            obst = np.zeros((3, 1))
            obst[0, 0] = local_x_y_angle[0, 0]
            obst[1, 0] = local_x_y_angle[0, 1]
            obst[2, 0] = local_x_y_angle[0, 2]
            obsSigma = np.zeros((3, 3))
            obsSigma[0, 0] = obsSigma_x
            obsSigma[1, 1] = obsSigma_y
            obsSigma[2, 2] = obsSigma_angle
            obsA = np.eye(3)
            obsb = np.array([[0], [0], [0]])
            (logC, nnmu, nnSigma) = gaussian_dist_yxxmu_product_x( \
                obst, obsA, obsb, obsSigma, nmu, nSigma)
        elif b_longitudinal_correction and not b_lateral_correction:
            obst = np.zeros((1, 1))
            obst[0, 0] = local_x_y_angle[0, 0]
            obsSigma = np.zeros((1, 1))
            obsSigma[0, 0] = obsSigma_x
            obsA = np.eye(1)
            obsb = np.array([[0]])
            nmuX = np.zeros((1, 1))
            nmuX[0, 0] = nmu[0, 0]
            nSigmaX = np.zeros((1, 1))
            nSigmaX[0, 0] = nSigma[0, 0]

            (logC, nnmu_x, nnSigma_x) = gaussian_dist_yxxmu_product_x( \
                obst, obsA, obsb, obsSigma, nmuX, nSigmaX)
            nnmu[0, 0] = nnmu_x[0, 0]
            nnSigma[0, 0] = nnSigma_x[0, 0]

        elif b_lateral_correction and not b_longitudinal_correction:
            obst = np.zeros((2, 1))
            obst[0, 0] = local_x_y_angle[0, 1]
            obst[1, 0] = local_x_y_angle[0, 2]
            obsSigma = np.zeros((2, 2))
            obsSigma[0, 0] = obsSigma_y
            obsSigma[1, 1] = obsSigma_angle
            obsA = np.eye(2)
            obsb = np.array([[0], [0]])
            nmu_y_angle = np.zeros((2, 1))
            nmu_y_angle[0, 0] = nmu[1, 0]
            nmu_y_angle[1, 0] = nmu[2, 0]
            nSigma_y_angle = np.zeros((2, 2))
            nSigma_y_angle[0:2, 0:2] = nSigma[1:3, 1:3]

            (logC, \
             nnmu_y_angle, \
             nnSigma_y_angle) = gaussian_dist_yxxmu_product_x(obst, obsA, obsb, obsSigma, \
                                                                 nmu_y_angle, nSigma_y_angle)
            nnmu[1, 0] = nnmu_y_angle[0, 0]
            nnmu[2, 0] = nnmu_y_angle[1, 0]
            nnSigma[1:3, 1:3] = nnSigma_y_angle
        else:
            # No correction, use predicted pose as updated pose
            logC = np.asarray([0.0])

        return nnmu, nnSigma, logC

    def updateStep(self, b_longitudinal_correction, b_lateral_correction, local_x_y_angle, nmu, nSigma):
        nnmu = np.copy(nmu)
        nnSigma = np.copy(nSigma)
        obst = np.zeros((3, 1))
        obsSigma = np.zeros((3, 3))
        obsSigma[0, 0] = obsSigma_x
        obsSigma[1, 1] = obsSigma_y
        obsSigma[2, 2] = obsSigma_angle
        obsb = np.array([[0.0], [0.0], [0.0]])

        # To align the angles into the same sign
        if local_x_y_angle[0, 2] < ( -np.pi / 2) and nmu[2,0] > (np.pi /2):
            local_x_y_angle[0, 2] = local_x_y_angle[0, 2] + 2 * np.pi
        elif local_x_y_angle[0, 2] > ( np.pi / 2) and nmu[2,0] < (-np.pi /2):
            local_x_y_angle[0, 2] = local_x_y_angle[0, 2] - 2 * np.pi



        if b_longitudinal_correction and b_lateral_correction:
            obst[0, 0] = local_x_y_angle[0, 0]
            obst[1, 0] = local_x_y_angle[0, 1]
            obst[2, 0] = local_x_y_angle[0, 2]

            obsA = np.eye(3)
            (logC, nnmu, nnSigma) = gaussian_dist_yxxmu_product_x( \
                obst, obsA, obsb, obsSigma, nmu, nSigma)
        elif b_longitudinal_correction and not b_lateral_correction:
            obst[0, 0] = local_x_y_angle[0, 0]
            obsA = np.asarray([[1.0, 0.0, 0.0],\
                               [0.0, 0.0, 0.0],\
                               [0.0, 0.0, 0.0]])

            (logC, nnmu, nnSigma) = gaussian_dist_yxxmu_product_x( \
                obst, obsA, obsb, obsSigma, nmu, nSigma)

        elif b_lateral_correction and not b_longitudinal_correction:
            obst[1, 0] = local_x_y_angle[0, 1]
            obst[2, 0] = local_x_y_angle[0, 2]
            obsA = np.asarray([[0.0, 0.0, 0.0],\
                               [0.0, 1.0, 0.0],\
                               [0.0, 0.0, 1.0]])
            (logC, nnmu, nnSigma) = gaussian_dist_yxxmu_product_x( \
                obst, obsA, obsb, obsSigma, nmu, nSigma)

        logC = np.asarray([0.0])
        return nnmu, nnSigma, logC

    def radarFilter1(self, options, frame_idx, gt_pos, mapgraph, mapdynamics, \
                    post_previous, local_radar_points, local_normals, yt):
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        debugNodes = set()

        # Inference variables
        Npts = 400 # number of MC samples to use when resorting to sampling
        #alphaThresh = 1e-2 # threshold at which a component is sampled
        compTransProbThresh = 1e-16
        compKalmanGradThresh = 1e-8 # default

        # Build a local points rtree for fast indexing
        local_points_rtree = build_rtree_points(local_radar_points)

        normalizeLogV = True

        weight_sum = 0.0
        stateDim = mapdynamics.state_dim
        localKeys = dict((vt, n) for (n, vt) in enumerate(mapgraph.nodes_iter()))
        allStreets = localKeys
        nKeys = len(localKeys)
        logVhatsumVec = np.empty(nKeys)
        postHat = MapGMMDistribution()
        all_segments = np.asarray(mapgraph.line_segment_list)
        local_normal_radians = np.arctan2(local_normals[1, :], local_normals[0, :])
        b_longitudinal_correction = longitudinal_correction(local_normal_radians)

        # Initialize a new GMM
        for vt in localKeys:
            if vt == SINK_ROAD:
                continue
            postHat.logV[vt] = 0.0
            postHat.Theta[vt] = GaussianMixtureModel(stateDim)
            postHat.Offset[vt] = GaussianMixtureModel(1)

        tmpt = time.time()

        for vt_1 in allStreets:
            if vt_1 == SINK_ROAD:
                continue
            curr_startt = time.time()
            if not np.isfinite(post_previous.logV[vt_1]):
                postHat.Info[vt_1] = {'filter_time':(time.time() - curr_startt)}
                continue
            prevThetaDist = post_previous.Theta[vt_1]
            prevOffsetDist = post_previous.Offset[vt_1]
            prevStreetLogV = post_previous.logV[vt_1]

            # accounting variables
            totalNComps = 0
            nTransComps = 0
            nMidComps = 0
            nDiscMidComps = 0

            intersection_offset = 0.2  # x / project_scale * 1000 (m)
            street_xlim = mapgraph.node[vt_1]['xlim']
            street_ylim = mapgraph.node[vt_1]['ylim']
            intersect_segments = np.array(list(mapgraph.rtreeIndex.intersection((street_xlim[0] - intersection_offset, \
                                                                                 street_ylim[0] - intersection_offset, \
                                                                                 street_xlim[1] + intersection_offset, \
                                                                                 street_ylim[1] + intersection_offset))))

            # print 'Number of segments in this street: {}'.format(len(intersect_segments))

            transDists = dict((vt,GaussianMixtureModel(stateDim)) for vt in mapgraph.successors_iter(vt_1))
            transDistsOffset = dict((vt,GaussianMixtureModel(1)) for vt in mapgraph.successors_iter(vt_1))

            # Per component on the street
            for i in range(prevThetaDist.numComps()):


                plogw = prevThetaDist.getLogW(i)
                if not np.isfinite(plogw):
                    continue


                print("\033[92m {}\033[00m".format(
                    '#######################################################################################################################'))
                pmu = prevThetaDist.getMu(i)
                pSigma = prevThetaDist.getSigma(i)
                sigmaPts = None
                pmuY = prevOffsetDist.getMu(i)
                pSigmaY = prevOffsetDist.getSigma(i)
                nmuOffset = pmuY

                prior_mean = predict_mean = post_mean = np.zeros((1, 5))
                prior_mean[0, 0:4] = pmu[0:4, 0]
                prior_mean[0, 4] = pmuY[0, 0]
                prior_mean_pose = trans_pose_global( mapgraph, vt_1, prior_mean)


                # Prediction
                trans_probs, \
                trans_prob_derivs,\
                vtIndices = mapdynamics.street_transition_probs(mapgraph, vt_1, pmu,
                                                                Sigma_st_1=pSigma,
                                                             computeDerivs=True)
                # pPropagate the process noise here
                nSigmaOffset = pSigmaY + (dv_offset_move_sigma * mapdynamics.dt) ** 2


                print 'prior pose = {0}'.format(prior_mean_pose)
                print ('vt_1: ' + vt_1)
                print ('vt_1 length: ' + str(mapgraph.node[vt_1]['length']))

                if debugFilter:
                    print '', i, np.exp(plogw), pmu.reshape(stateDim)
                for (vt, transI) in vtIndices.iteritems():
                    if vt is not vt_1:
                        totalNComps += 1
                    if vt not in localKeys or vt == SINK_ROAD:
                        continue

                    vtlen = mapgraph.node[vt]['length']
                    (obsA, obsb, (obsOdomSigma, obsCholSigma)) = mapdynamics.observation_distribution(mapgraph, vt,
                                                                                                  returnChol=True)
                    (moveA, moveb, moveSigma) = mapdynamics.state_transition_distribution(mapgraph, vt, vt_1)
                    trans_prob_street = trans_probs[0, transI]
                    dtrans_prob_street = np.sum(np.power(trans_prob_derivs[:, transI], 2.0))

                    if debugFilter:
                        print("\033[91m {}\033[00m".format(
                            '------------------------------------------------------------------------------------------------------------'))
                        print ('vt: ' + vt)
                        print ('vt length: ' + str(vtlen))
                        print('compTransProbThresh: ' + str(compTransProbThresh))
                        print ('trans_prob_street: ' + str(trans_prob_street))
                        print('compKalmanGradThresh: ' + str(compKalmanGradThresh))
                        print ('dtrans_prob_street: ' + str(dtrans_prob_street))

                    if trans_prob_street > compTransProbThresh and dtrans_prob_street < compKalmanGradThresh:
                        # This mode will always make the transition
                        cleanAdd = True
                        if vt is not vt_1:
                            nTransComps += 1
                        nmu = np.dot(moveA, pmu) + moveb
                        nSigma = moveSigma + np.dot(moveA, np.dot(pSigma, moveA.T))
                        nlogw = plogw + np.log(trans_prob_street)
                        if debugFilter:
                            print 'This mode will always make the transition'

                    elif trans_prob_street > compTransProbThresh:
                        if debugFilter:
                            print 'This mode may or may not make the transition, resort to sampling'
                        cleanAdd = False
                        if vt is not vt_1:
                            nMidComps += 1
                        # This mode may or may not make the transition, resort to sampling
                        if sigmaPts is None:
                            sigmaPts = pmu + np.dot(np.linalg.cholesky(pSigma), np.random.randn(stateDim, Npts))
                            sigmaPtInds = mapdynamics.street_state_mask(mapgraph, sigmaPts, vt_1)
                            sigmaPts = (sigmaPts[:, sigmaPtInds]).reshape((stateDim, -1))
                            currNPts = sigmaPts.shape[1]
                            if currNPts > 0:
                                logpSigmaPts = mapdynamics.street_transition_logprob(mapgraph, None, vt_1, sigmaPts)

                        if debugFilter:
                            print('currNPts: '+ str(currNPts))

                        if currNPts == 0:
                            continue
                        logp = logpSigmaPts[:, transI]
                        logsumexpp = logsumexp(logp)

                        if debugFilter:
                            print('logsumexpp: '+ str(logsumexpp))
                        if not np.isfinite(logsumexpp):
                            if vt is not vt_1:
                                nDiscMidComps += 1
                            continue

                        movePts = moveb + np.dot(moveA, sigmaPts)
                        movePtWs = np.exp(logp - logsumexpp)

                        nlogw = plogw + logsumexpp - np.log(currNPts)
                        nmu = np.dot(movePts, movePtWs.reshape((currNPts, 1))).reshape(stateDim, 1)
                        nSigma = moveSigma + np.dot(movePts - nmu,
                                                    np.multiply(movePtWs.reshape((1, currNPts)), movePts - nmu).T)
                    else:
                        # This mode will never make the transition
                        continue



                    predict_mean[0, 0:4] = nmu[0:4, 0];
                    predict_mean[0, 4] = nmuOffset[0, 0]
                    predict_mean_pose = trans_pose_global( mapgraph, vt, predict_mean)

                    # Update step
                    if np.isfinite(nlogw):
                        (logC, nnmu, nnSigma, nnmuOffset, nnSigmaOffset) = self.updateSampling(options, \
                                                                                mapgraph, vt_1, \
                                                                                yt, obsA, obsb, obsOdomSigma,\
                                                                 nmu, nSigma, nmuOffset, nSigmaOffset, \
                            b_longitudinal_correction, local_radar_points, local_points_rtree, local_normal_radians,\
                                                    all_segments, intersect_segments)

                        nnlogw = post_previous.logV[vt_1] + nlogw + logC


                        post_mean[0, 0:4] = nnmu[0:4, 0];
                        post_mean[0, 4] = nnmuOffset[0, 0]
                        post_mean_pose = trans_pose_global( mapgraph, vt, post_mean)
                        if debugFilter:
                            print 'nnmu = {0}, posterior mean = {1}'.format(post_mean, \
                                                                            post_mean_pose)


                        if debugPredictModel and (frame_idx) >= 5000:
                            n_display_sample = 200
                            prior_thetas = self.gaussian_sampling(True, pmu, pmuY, pSigma, pSigmaY,n_display_sample)
                            prior_poses = trans_pose_global( mapgraph, vt_1, prior_thetas)
                            predict_thetas = self.gaussian_sampling(True, nmu, nmuOffset, nSigma, nSigmaOffset,n_display_sample)
                            predict_poses = trans_pose_global( mapgraph, vt, predict_thetas)
                            post_thetas = self.gaussian_sampling(True, nnmu, nnmuOffset, nnSigma, nnSigmaOffset,n_display_sample)
                            posterior_poses = trans_pose_global( mapgraph, vt, post_thetas)

                            fig_dist = plt.figure(figsize=displayFigSz)
                            ax_distributions = fig_dist.add_subplot(111)
                            lines = []
                            street_node_vt_1 = mapgraph.node[vt_1]
                            ax_distributions.plot([street_node_vt_1['origin'][0], street_node_vt_1['terminus'][0]], \
                                                       [street_node_vt_1['origin'][1], street_node_vt_1['terminus'][1]], \
                                                       linewidth=2.0, c='black')

                            intersect_segments_idx = np.array(
                                list(
                                    mapgraph.rtreeIndex.intersection((street_node_vt_1['xlim'][0] - intersection_offset, \
                                                                      street_node_vt_1['ylim'][0] - intersection_offset, \
                                                                      street_node_vt_1['xlim'][1] + intersection_offset, \
                                                                      street_node_vt_1['ylim'][1] + intersection_offset))))
                            for l_idx in range(intersect_segments_idx.shape[0]):
                                line = all_segments[intersect_segments_idx[l_idx]]
                                lines.append(np.vstack((line[0:2], line[2:4])))
                            ax_distributions.add_collection(
                                LineCollection(lines, transOffset=ax_distributions.transData, linewidths=0.1,
                                               colors='green'))

                            ax_distributions.scatter(prior_mean_pose[0,0], prior_mean_pose[0,1], s=2.0, c='black',
                                                     label='prior mean:' + str(prior_mean_pose[0,0]), marker='D')
                            ax_distributions.scatter(predict_mean_pose[0,0], predict_mean_pose[0,1], s=2.0, c='blue',
                                                     label='predict mean:' + str(predict_mean_pose[0,0]), marker='D')
                            ax_distributions.scatter(post_mean_pose[0,0], post_mean_pose[0,1], s=2.0, c='green',
                                                     label='updated mean:' + str(post_mean_pose[0,0]), marker='D')

                            ax_distributions.scatter(prior_poses[:, 0], prior_poses[:, 1], s=0.2, c='black', \
                                                          label='prior samples', alpha=0.2)
                            ax_distributions.scatter(predict_poses[:, 0], predict_poses[:, 1], s=0.2, c='blue', \
                                                          label='predict samples', alpha=0.2)
                            ax_distributions.scatter(posterior_poses[:, 0], posterior_poses[:, 1], s=0.2, c='green', \
                                                          label='updated samples', alpha=0.2)

                            # ax_distributions.quiver(prior_poses[:, 0], prior_poses[:, 1], \
                            #                              np.cos(prior_poses[:, 2]), np.sin(prior_poses[:, 2]),
                            #                              width=0.002, color='black', alpha=0.2)
                            ax_distributions.quiver(predict_poses[:, 0], predict_poses[:, 1], \
                                                         np.cos(predict_poses[:, 2]), np.sin(predict_poses[:, 2]),
                                                         width=0.002, color='blue', alpha=0.2)
                            ax_distributions.quiver(posterior_poses[:, 0], posterior_poses[:, 1], \
                                                         np.cos(posterior_poses[:, 2]), np.sin(posterior_poses[:, 2]),
                                                         width=0.002, color='green', alpha=0.2)

                            ax_distributions.plot([mapgraph.node[vt]['origin'][0], mapgraph.node[vt]['terminus'][0]], \
                                                  [mapgraph.node[vt]['origin'][1], mapgraph.node[vt]['terminus'][1]], \
                                                  linewidth=1.0, c='blue')
                            ax_distributions.scatter(gt_pos[0], gt_pos[1], s=0.2, c='red', label='gt') # gt

                            ax_distributions.legend()
                            ax_distributions.set_title('frame idx: '+str(frame_idx) + \
                                                       ', street w: ' + str(np.exp(prevStreetLogV))[0:5] + \
                                                       ', trans prob street: ' + str(trans_prob_street)[0:5] + \
                                                       ' \nprior w: ' + str(np.exp(plogw))[0:5] + \
                                                       ', predict w: ' + str(np.exp(nlogw))[0:5] + \
                                                       ', likelihood: ' + str(np.exp(logC[0]))[0:5] + \
                                                       ', update w: ' + str(np.exp(nnlogw[0]))[0:5] + \
                                                            '\nvt_1:' + vt_1 + '\nvt  :' + vt)
                            ax_distributions.set_xlim(
                                [gt_pos[0] - displayOffset, gt_pos[0] + displayOffset])
                            ax_distributions.set_ylim(
                                [gt_pos[1] - displayOffset, gt_pos[1] + displayOffset])
                            directory = cwd + '/results/visualization/' + options.dataname + '/prior_predict_post_' + \
                                        str(frame_idx).zfill(6) + '_'
                            fig_name = directory + str(np.exp(nnlogw[0]))[0:8] + '.png'
                            fig_dist.savefig(fig_name, dpi=100)


                        if debugFilter:
                            print 'plogw = {0}, nlogw = {1}, nnlogw = {2}, logC = {3}'.format(plogw,\
                                                        nlogw, nnlogw, logC)

                        if np.isfinite(nnlogw):
                            cdfVals = stats.norm.cdf(np.array([-streetMargin,vtlen+streetMargin]),loc = nnmu[0], scale = np.sqrt(nnSigma[0,0]))
                            if cdfVals[1] - cdfVals[0] >= streetMarginThresh:
                                if cleanAdd or vt == vt_1:
                                    if debugFilter:
                                        print 'clean add'
                                    postHat.Theta[vt].addComponentLogW(nnlogw, nnmu, nnSigma)
                                    postHat.Offset[vt].addComponentLogW(nnlogw, nnmuOffset, nnSigmaOffset)
                                else:
                                    if debugFilter:
                                        print 'mixed add'
                                    transDists[vt].addComponentLogW(nnlogw, nnmu, nnSigma)
                                    transDistsOffset[vt].addComponentLogW(nnlogw, nnmuOffset, nnSigmaOffset)
                            elif debugFilter:
                                print 'dropped cdf'
                        elif debugFilter:
                            print 'dropped inf logC'
                    else:
                        if debugFilter:
                            print 'plogw = {0}, nlogw = {1}'.format(plogw,nlogw)
                            print 'prior mean = {0}, predict mean = {1}'.format(prior_mean_pose, predict_mean_pose)

            if debugFilter:
                print ''
            for vt in mapgraph.successors_iter(vt_1):
                if vt == SINK_ROAD or transDists[vt].numComps() == 0:
                    continue
                if vt == vt_1:
                    postHat.Theta[vt].addMixtureLogW(0, transDists[vt])
                else:
                    cvtlogW = transDists[vt].normalizeWeights()
                    (nnmu,nnSigma) = transDists[vt].computeMeanCovar()
                    postHat.Theta[vt].addComponentLogW(cvtlogW, nnmu, nnSigma)
                    (nnmuOffset,nnSigmaOffset) = transDistsOffset[vt].computeMeanCovar()
                    postHat.Offset[vt].addComponentLogW(cvtlogW, nnmuOffset, nnSigmaOffset)
                    if debugFilter:
                        print 'mixed added: a new component to the successor vt street'
                        new_mean = np.zeros((1, 5))
                        new_mean[0, 0:4] = nnmu[0:4, 0]
                        new_mean[0, 4] = nnmuOffset[0, 0]
                        new_pose = trans_pose_global( mapgraph, vt_1, new_mean)
                        print 'new component pose = {0}, nnmu = {1}, exp(cvtlogW) = {2}'.format(new_pose, \
                                                                                nnmu, np.exp(cvtlogW))

            postHat.Info[vt_1] = {'filter_time':(time.time() - curr_startt)}
        if debugFilter:
            print 'Radar filtering done in {0}s.'.format(time.time() - tmpt)

        for (vt,cvtInd) in localKeys.iteritems():
            # debugFilter = vt in debugNodes and vt in localKeys
            if vt == SINK_ROAD:
                logVhatsumVec[cvtInd] = -np.inf
                continue

            cThetahat = postHat.Theta[vt]
            if cThetahat.numComps() > 0:
                clogVhatsum = postHat.Theta[vt].normalizeWeights()
            else:
                clogVhatsum = -np.inf

            # if debugFilter:
            #     print 'Normalization {0}: NumComps {1}'.format(vt,cThetahat.numComps())
            #     print '  logVhatsum = {0}'.format(clogVhatsum)

            if np.isfinite(clogVhatsum):
                postHat.logV[vt] = clogVhatsum
                logVhatsumVec[cvtInd] = clogVhatsum
            else:
                logVhatsumVec[cvtInd] = -np.inf
                postHat.logV[vt] = -np.inf
                postHat.Theta[vt] = ()
        logVhatsum = logsumexp(logVhatsumVec)

        if normalizeLogV:
            # If we're looking at the whole distribution, normalize it.
            for vt in mapgraph.nodes_iter():
                if vt == SINK_ROAD:
                    continue
                # debugFilter = vt in debugNodes and vt in localKeys
                # if debugFilter:
                #     print '{0}: logV = {1}, logVhatsum = {2}'.format(vt,postHat.logV[vt],logVhatsum)
                postHat.logV[vt] -= logVhatsum
                logDiscardCompThresh = -50.0 + (np.log(mapgraph.node[vt]['length']) - np.log(mapgraph.totalLength))
                if np.isfinite(postHat.logV[vt]) and postHat.logV[vt] <= logDiscardCompThresh:
                    postHat.logV[vt] = -np.inf
                    postHat.Theta[vt] = ()
                    if debugFilter:
                        print '  logV = {1} < {2}, dropping street'.format(vt,postHat.logV[vt],logDiscardCompThresh)

        # Todo: Visualize all the means of the mixture models
        fig = plt.figure(figsize=displayFigSz)
        ax = fig.add_subplot(111)
        ax.scatter(gt_pos[0], gt_pos[1], s=0.2, c='r', label='gt')
        intersection_offset = 0.2  # 0.2 / project_scale * 1000 (m)

        for vt in allStreets:
            if vt == SINK_ROAD:
                continue
            if not np.isfinite(postHat.logV[vt]):
                continue
            ThetaDist = postHat.Theta[vt]
            OffsetDist = postHat.Offset[vt]
            LogVstreet = postHat.logV[vt]
            logWeightSumStreet = np.log(np.sum(np.exp(ThetaDist.getLogWs())))

            for i in range(ThetaDist.numComps()):
                logw = ThetaDist.getLogW(i)
                mu = ThetaDist.getMu(i)
                muOffset = OffsetDist.getMu(i)
                theta = np.vstack((mu, muOffset)).T
                normalized_w = np.exp(logw - logWeightSumStreet + LogVstreet)

                # Remove the component with small weight
                if normalized_w < SMALL_COMPONENT_THRESH:
                    postHat.Theta[vt].removeComp(i)
                    postHat.Offset[vt].removeComp(i)
                    if debugFilter:
                        print 'Removed component with weight: {0}'.format(normalized_w)
                    continue

                curr_pos = trans_pose_global( mapgraph, vt, theta)[0]
                lines = []
                ax.plot([mapgraph.node[vt]['origin'][0], mapgraph.node[vt]['terminus'][0]], \
                        [mapgraph.node[vt]['origin'][1], mapgraph.node[vt]['terminus'][1]], \
                        linewidth=0.3, c='black')
                street_xlim = mapgraph.node[vt]['xlim']
                street_ylim = mapgraph.node[vt]['ylim']
                intersect_segments_idx = np.array(
                    list(mapgraph.rtreeIndex.intersection((street_xlim[0] - intersection_offset, \
                                                           street_ylim[0] - intersection_offset, \
                                                           street_xlim[1] + intersection_offset, \
                                                           street_ylim[1] + intersection_offset))))
                for l_idx in range(intersect_segments_idx.shape[0]):
                    line = all_segments[intersect_segments_idx[l_idx]]
                    lines.append(np.vstack((line[0:2], line[2:4])))
                ax.add_collection(
                    LineCollection(lines, transOffset=ax.transData, linewidths=0.1, colors='green'))
                ax.scatter(curr_pos[0], curr_pos[1], s=0.2, c='b', alpha=normalized_w)
                ax.quiver(curr_pos[0], curr_pos[1], np.cos(curr_pos[2]), np.sin(curr_pos[2]), \
                          width=0.002, color='b', alpha=normalized_w)


                # draw_pose(options, frame_idx, compId, mapgraph, vt, gt_pos,\
                #           theta, w, intersect_segments, all_segments)
                # compId += 1
                # print 'logw = {0}, logweightsumStreet = {1}, streetV = {2} '.format(np.exp(logw),
                #                                                                     np.exp(logWeightSumStreet),
                #                                                                     np.exp(LogVstreet))
        ax.legend()
        ax.set_title('frame idx: ' + str(frame_idx))
        ax.set_xlim([gt_pos[0] - intersection_offset, gt_pos[0] + intersection_offset])
        ax.set_ylim([gt_pos[1] - intersection_offset, gt_pos[1] + intersection_offset])
        directory = cwd + '/results/visualization/' + options.dataname + '/'
        fig_name = directory + str(frame_idx).zfill(6) + '_' 'posterior.png'
        fig.savefig(fig_name, dpi=100)


        # # Todo: Visualize some samples
        # if debugFilter:
        #     max_weight = 0.0
        #     max_number_of_points = 0
        #     max_w_idx = -1
        #     max_n_idx = -1
        #     for idx, w in enumerate(self.weight_list):
        #         weight_sum += w
        #         if w > max_weight:
        #             max_weight = w
        #             max_w_idx = idx
        #         points_weight = self.all_points_weights[idx]
        #         survived_idx = np.where(points_weight > 0)[0]
        #         number_of_points = survived_idx.shape[0]
        #         if number_of_points > max_number_of_points:
        #             max_number_of_points = number_of_points
        #             max_n_idx = idx
        #     max_weight = max_weight / weight_sum
        #     indexes = [max_w_idx, max_n_idx]
        #     for idx in indexes:
        #         fig = plt.figure(figsize=displayFigSz)
        #         ax = fig.add_subplot(111)
        #         intersect_segments = self.intersect_segment_list[idx]
        #         global_position = self.pos_list[idx]
        #         heading = self.heading_list[idx]
        #         weight = self.weight_list[idx] / weight_sum
        #         points = self.points_global_list[idx]
        #         points_weights = self.all_points_weights[idx]
        #         street_curr = self.streets[idx]
        #         ax = draw_points_lines(ax, street_curr, gt_pos, global_position, heading, weight,
        #                                max_weight, max_number_of_points, points, points_weights,\
        #                                local_normals, intersect_segments, all_segments)
        #         ax.set_xlim([gt_pos[0] - 0.2, gt_pos[0] + 0.2])
        #         ax.set_ylim([gt_pos[1] - 0.2, gt_pos[1] + 0.2])
        #         directory = cwd + '/results/initial_result/samples_' + str(frame_idx).zfill(6)
        #         fig_name = directory + '_' + str(idx) + '.png'
        #         fig.savefig(fig_name, dpi=100)
        #         plt.close('all')

        self.weight_list = []
        self.points_global_list = []
        self.all_points_weights = []
        self.pos_list = []
        self.heading_list = []
        self.intersect_segment_list = []
        self.streets = []
        self.counter += 1

        if debugFilter:
            print ''
        return (postHat,logVhatsum)



    def radarFilter3(self, options, frame_idx, gt_pos, mapgraph, mapdynamics, \
                    post_previous, local_radar_points, local_normals, yt, odom_y):
        tmpt = time.time()


        # draw all the features and points
        if drawFeatures:    
            fig_features = plt.figure(figsize=(5, 5))
            ax_features = fig_features.add_subplot(111)
            ax_features.scatter(local_radar_points[0, :], local_radar_points[1, :], c='black', s=0.2)
            # ax_features.quiver(local_radar_points[0, :],  local_radar_points[1, :],\
            #         np.cos(local_normals), np.sin(local_normals), \
            #         width=0.002, color='blue', alpha=0.8)
            ax_features.set_xlim([-100, 100])
            ax_features.set_ylim([-100, 100])
            directory = cwd + '/results/visualization/' + options.dataname + \
                    '/points' + str(frame_idx).zfill(4) 
            fig_feature_name = directory +  '.png'
            fig_features.savefig(fig_feature_name, dpi=displayDPI)


        # scale the points to km
        local_radar_points[0:2,:] = local_radar_points[0:2,:] / 1000.0 # km


        # Filter variables
        odom_x_angle = yt * mapdynamics.dt
        uk = np.asarray([odom_x_angle[0,0], odom_y, odom_x_angle[1,0]])
        Qk = np.asarray([[moveVarX,          0.0,            0.0],\
                         [     0.0,     moveVarY,            0.0],\
                         [     0.0,          0.0,   moveVarAngle]])

        # print 'uk = {0}'.format(uk)


        # To avoid radar odometry degeneracy
        if self.initialized:
            curr_x_velocity = odom_x_angle[0,0]
            prev_x_velocity = self.last_uk[0] / mapdynamics.dt
            x_acceleration = (curr_x_velocity - prev_x_velocity) / mapdynamics.dt
            # print 'x acceleration = {0}'.format(x_acceleration)
        else:
            self.last_uk = uk
            self.initialized = True




        # Inference variables
        compTransProbThresh = 0.005
        intersection_offset = 0.2  # x / project_scale * 1000 (m)
        new_transition_streets = []
        new_transition_streets = []
        num_comp_transited = 0 # number of components that transited successfully
        reset_transition_streets = [] # potential streets that can be transited on when it is lost due to drift on longitutude direction.

        # # Build a local points rtree for fast indexing
        # local_points_rtree = build_rtree_points(local_radar_points)
        normalizeLogV = True

        stateDim = mapdynamics.state_dim
        localKeys = dict((vt, n) for (n, vt) in enumerate(mapgraph.nodes_iter()))
        allStreets = localKeys
        nKeys = len(localKeys)
        logVhatsumVec = np.empty(nKeys)
        postHat = MapGMMDistribution()
        all_segments = np.asarray(mapgraph.line_segment_list)
        local_normal_radians = np.arctan2(local_normals[1, :], local_normals[0, :])

        global_pose_obst, \
        global_pose_sigma, \
        b_longitudinal_correction, \
        b_lateral_correction = self.global_observation(options, frame_idx, mapgraph, mapdynamics,\
                                                       post_previous, allStreets, odom_x_angle, odom_y,\
                                                       local_radar_points, local_normal_radians,\
                                                         all_segments, gt_pos)

        # Initialize a new GMM
        for vt in localKeys:
            if vt == SINK_ROAD:
                continue
            postHat.logV[vt] = 0.0
            postHat.Theta[vt] = GaussianMixtureModel(stateDim)


        for vt_1 in allStreets:
            if vt_1 == SINK_ROAD:
                continue
            curr_startt = time.time()
            if not np.isfinite(post_previous.logV[vt_1]):
                postHat.Info[vt_1] = {'filter_time':(time.time() - curr_startt)}
                continue
            prevThetaDist = post_previous.Theta[vt_1]
            prevStreetLogV = post_previous.logV[vt_1]


            obsA, obsb, Sigma_odom_x_angle = mapdynamics.observation_distribution(mapgraph, vt_1)

            # Per component on the street
            for i in range(prevThetaDist.numComps()):
                plogw = prevThetaDist.getLogW(i)
                if not np.isfinite(plogw):
                    continue
                # print("\033[92m {}\033[00m".format(
                #     '###################################### For each component ###################################'))
                if debugFilter:
                    print '###################################### For each component ###################################'
                pmu = prevThetaDist.getMu(i)
                pSigma = prevThetaDist.getSigma(i)

                # # Skip this street
                # if abs(pmu[2,0]) > (np.pi/2):
                #     print 'skip this street due to huge heading difference'
                #     continue

                prior_mean = predict_mean = post_mean = np.zeros((1, 3))
                prior_mean[0, :] = pmu[:, 0]
                prior_mean_pose = trans_pose_global( mapgraph, vt_1, prior_mean)

                # Prediction
                trans_probs, \
                trans_prob_derivs,\
                vtIndices = mapdynamics.street_transition_probs_with_odom(frame_idx, mapgraph, vt_1,\
                                                                          pmu, pSigma, \
                                                                          odom_x_angle, odom_y, \
                                                                          Sigma_odom_x_angle, \
                                                                          computeDerivs=True)
                if debugFilter:
                    print '', i, np.exp(plogw), pmu.reshape(stateDim)
                    print 'prior pose = {0}'.format(prior_mean_pose)
                    print ('vt-1: ' + vt_1)
                    print ('vt-1 length: ' + str(mapgraph.node[vt_1]['length']))
                    print ('vt-1 origin heading:' + str(mapgraph.get_road_heading(vt_1, [0.0])))
                    print 'number of next transition street = {0}'.format(len(vtIndices))
                    # print 'trans_probs:'
                    # print trans_probs
                for (vt, transI) in vtIndices.iteritems():
                    if vt not in localKeys or vt == SINK_ROAD:
                        # if vt == SINK_ROAD:
                            # print 'vt = {0} is SINK_ROAD'.format(vt)
                        continue

                    vtlen = mapgraph.node[vt]['length']
                    trans_prob_street = trans_probs[0, transI]

                    if debugFilter:
                        print("\033[91m {}\033[00m".format(
                            '--------------------------- The successor street vt --------------------------------'))
                        print ('vt: ' + vt)
                        print ('vt length: ' + str(vtlen)),
                        print ('vt origin heading:' + str(mapgraph.get_road_heading(vt, [0.0])))
                        # print ('vt end heading:' + str(mapgraph.get_road_heading(vt, [vtlen])))
                        print ('trans_prob_street: ' + str(trans_prob_street))


                    # if trans_prob_street > compTransProbThresh and dtrans_prob_street < compKalmanGradThresh:
                    if trans_prob_street > compTransProbThresh:
                        # This mode will always make the transition
                        cleanAdd = True

                        # Predict state onto new street
                        nmu, nSigma = transition_new_street(pmu, pSigma, uk, Qk,\
                                                               vt_1, vt, mapgraph)
                        if debugFilter:
                            print 'nmu = {0}'.format(nmu.reshape(stateDim))
                        vt_mu_sigma = {}
                        vt_mu_sigma['vt'] = vt
                        vt_mu_sigma['mu'] = nmu
                        vt_mu_sigma['sigma'] = nSigma
                        
                        # if frame_idx >= debugFrame:
                        #     print ('vt-1: ' + vt_1) 
                        #     print 'vt-1 origin = {0}'.format(mapgraph.node[vt_1]['origin'])
                        #     print 'vt-1 terminus = {0}'.format(mapgraph.node[vt_1]['terminus'])
                        #     print ('vt: ' + vt)
                        #     print 'vt origin = {0}'.format(mapgraph.node[vt]['origin'])
                        #     print 'vt terminus = {0}'.format(mapgraph.node[vt]['terminus'])
                        #     print ''
                            
                        street_origin_heading = mapgraph.get_road_heading(vt, [0.0])[0]
                        street_terminus_heading = mapgraph.get_road_heading(vt, [vtlen])[0]
                        if street_origin_heading == street_terminus_heading:
                            # line
                            if heading_thres < abs(nmu[2, 0]) < (np.pi * 2 - heading_thres):
                                reset_transition_streets.append(vt_mu_sigma)
                                if debugFilter:
                                    print 'skip this new line transition street due to huge heading difference'
                                continue
                        else:
                            # curve
                            if heading_thres < abs(nmu[2, 0]) < (np.pi * 2 - heading_thres) \
                                and  heading_thres < abs(nmu[2, 0] + street_origin_heading - street_terminus_heading)  < (np.pi * 2 - heading_thres)  :
                                reset_transition_streets.append(vt_mu_sigma)
                                if debugFilter :
                                    print 'skip this new curve transition street due to huge heading difference'
                                continue

                        predict_global_pose_vt = trans_pose_global( mapgraph, vt, nmu.reshape((1,3)))
                        nlogw = plogw + np.log(trans_prob_street)
                        num_comp_transited += 1

                    else:
                        # This mode will never make the transition
                        if debugFilter:
                            print 'This mode on {0} will never make the transition'.format(vt)
                            print ''
                        continue

                    predict_mean[0,:] = nmu[:, 0];
                    predict_mean_pose = trans_pose_global( mapgraph, vt, predict_mean)

                    # Update step
                    if np.isfinite(nlogw):
                        if b_longitudinal_correction or b_lateral_correction:
                            local_x_y_angle = global_pose_to_local_pose(mapgraph, vt, global_pose_obst, \
                                                                                       options, frame_idx, gt_pos)

                            nnmu, nnSigma, logC = self.updateStep(b_longitudinal_correction,\
                                                            b_lateral_correction, local_x_y_angle, nmu, nSigma)
                            if debugFilter:
                                print 'local_x_y_angle = {0}'.format(local_x_y_angle)
                                print 'nnmu = {0}'.format(nnmu.reshape(stateDim))
                        else:
                            nnmu = np.copy(nmu)
                            nnSigma = np.copy(nSigma)
                            logC = np.asarray([0.0])

                        nnlogw = post_previous.logV[vt_1] + nlogw + logC
                        post_mean[0, :] = nnmu[:, 0];
                        post_mean_pose = trans_pose_global( mapgraph, vt, post_mean)

                        if debugPredictModel and (frame_idx >= 9):
                            n_display_sample = 100
                            prior_thetas = self.gaussian_sampling(True, pmu, pSigma, n_display_sample)
                            prior_poses = trans_pose_global( mapgraph, vt_1, prior_thetas)
                            predict_thetas = self.gaussian_sampling(True, nmu, nSigma, n_display_sample)
                            predict_poses = trans_pose_global( mapgraph, vt, predict_thetas)
                            post_thetas = self.gaussian_sampling(True, nnmu, nnSigma, n_display_sample)
                            posterior_poses = trans_pose_global( mapgraph, vt, post_thetas)

                            fig_dist = plt.figure(figsize=displayFigSz)
                            ax_distributions = fig_dist.add_subplot(111)
                            lines = []
                            street_node_vt_1 = mapgraph.node[vt_1]
                            ax_distributions.plot([street_node_vt_1['origin'][0], street_node_vt_1['terminus'][0]], \
                                                       [street_node_vt_1['origin'][1], street_node_vt_1['terminus'][1]], \
                                                       linewidth=2.0, c='black')

                            intersect_segments_idx = np.array(
                                list(
                                    mapgraph.rtreeIndex.intersection((street_node_vt_1['xlim'][0] - intersection_offset, \
                                                                      street_node_vt_1['ylim'][0] - intersection_offset, \
                                                                      street_node_vt_1['xlim'][1] + intersection_offset, \
                                                                      street_node_vt_1['ylim'][1] + intersection_offset))))
                            for l_idx in range(intersect_segments_idx.shape[0]):
                                line = all_segments[intersect_segments_idx[l_idx]]
                                lines.append(np.vstack((line[0:2], line[2:4])))
                            ax_distributions.add_collection(
                                LineCollection(lines, transOffset=ax_distributions.transData, linewidths=0.1,
                                               colors='green'))


                            ax_distributions.scatter(prior_mean_pose[0,0], prior_mean_pose[0,1], s=2.0, c='black',
                                                     label='prior mean:' + str(prior_mean_pose[0,0]), marker='D')
                            # ax_distributions.scatter(predict_global_pose_vt_1[0,0], predict_global_pose_vt_1[0,1],\
                            #                          s=2.0, c='blue', \
                            #                          label='predict mean vt-1:' + str(predict_mean_pose[0,0]), marker='D')
                            ax_distributions.scatter(predict_global_pose_vt[0,0], predict_global_pose_vt[0,1],\
                                                     s=4.0, c='yellow', \
                                                     label='predict mean vt:' + str(predict_mean_pose[0,0]), marker='D')
                            ax_distributions.scatter(post_mean_pose[0,0], post_mean_pose[0,1], s=2.0, c='green',
                                                     label='updated mean:' + str(post_mean_pose[0,0]), marker='D')

                            # ax_distributions.scatter(prior_poses[:, 0], prior_poses[:, 1], s=0.2, c='black', \
                            #                               label='prior samples', alpha=0.1)
                            ax_distributions.scatter(predict_poses[:, 0], predict_poses[:, 1], s=0.2, c='blue', \
                                                          label='predict samples', alpha=0.1)
                            # ax_distributions.scatter(posterior_poses[:, 0], posterior_poses[:, 1], s=0.2, c='green', \
                            #                               label='updated samples', alpha=0.1)

                            # ax_distributions.quiver(predict_poses[0:-1:3,0], predict_poses[0:-1:3,1], \
                            #                         np.cos(predict_poses[0:-1:3,2]), np.sin(predict_poses[0:-1:3,2]), \
                            #           width=0.002, color='blue', alpha=0.2)

                            ax_distributions.plot([mapgraph.node[vt]['origin'][0], mapgraph.node[vt]['terminus'][0]], \
                                                  [mapgraph.node[vt]['origin'][1], mapgraph.node[vt]['terminus'][1]], \
                                                  linewidth=1.0, c='blue')
                            ax_distributions.scatter(gt_pos[0], gt_pos[1], s=0.2, c='red', label='gt') # gt
                            ax_distributions.scatter(global_pose_obst[0], global_pose_obst[1], s=0.2, c='purple', \
                                                     label='global obst')
                            ax_distributions.quiver(global_pose_obst[0], global_pose_obst[1], \
                                                    np.cos(global_pose_obst[2]), np.sin(global_pose_obst[2]), \
                                      width=0.002, color='purple', alpha=0.5)
                            street_node  =  mapgraph.node[vt]
                            ax_distributions.scatter(street_node['origin'][0], street_node['origin'][1], s=5, c='g', label='origin',
                                       alpha=0.5)
                            ax_distributions.legend()
                            ax_distributions.set_title('frame idx: '+str(frame_idx) + \
                                                       ', street w: ' + str(np.exp(prevStreetLogV))[0:5] + \
                                                       ', trans prob street: ' + str(trans_prob_street)[0:5] + \
                                                       ' \nprior w: ' + str(np.exp(plogw))[0:5] + \
                                                       ', predict w: ' + str(np.exp(nlogw))[0:5] + \
                                                       ', likelihood: ' + str(np.exp(logC[0]))[0:5] + \
                                                       ', update w: ' + str(np.exp(nnlogw[0]))[0:5] + \
                                                            '\nvt_1:' + str(vt_1) + '\nvt  :' + str(vt))

                            ax_distributions.set_xlim(
                                [gt_pos[0] - displayOffset, gt_pos[0] + displayOffset])
                            ax_distributions.set_ylim(
                                [gt_pos[1] - displayOffset, gt_pos[1] + displayOffset])
                            directory = cwd + '/results/visualization/' + options.dataname + '/prior_predict_post_' + \
                                        str(frame_idx).zfill(6) + '_'
                            fig_name = directory + str(np.exp(nnlogw[0]))[0:8] + '.png'
                            fig_dist.savefig(fig_name, dpi=100)

                        if np.isfinite(nnlogw):
                            if cleanAdd:
                                if debugFilter:
                                    print 'clean add'
                                if vt == vt_1:
                                    postHat.Theta[vt].removeOldComp()
                                    postHat.Theta[vt].addComponentLogW(nnlogw, nnmu, nnSigma)
                                else:
                                       postHat.Theta[vt].addComponentLogW(nnlogw, nnmu, nnSigma)
                                if vt not in new_transition_streets:
                                    new_transition_streets.append(vt)
                        elif debugFilter:
                            print 'dropped inf logC'
                    else:
                        if debugFilter:
                            print 'plogw = {0}, nlogw = {1}'.format(plogw,nlogw)
                            print 'prior mean = {0}, predict mean = {1}'.format(prior_mean_pose, predict_mean_pose)

            if debugFilter:
                print ''

            postHat.Info[vt_1] = {'filter_time':(time.time() - curr_startt)}


        # Check if we have zero componnet
        if num_comp_transited == 0:
            print 'Reseting due to zero component.'
            for vt_mu_sigma in reset_transition_streets:
                vt = vt_mu_sigma['vt']
                mu = vt_mu_sigma['mu']
                sigma = vt_mu_sigma['sigma']
                postHat.Theta[vt].addComponentLogW(0.0, mu, sigma)
                if debugFilter:
                    print vt

        for (vt,cvtInd) in localKeys.iteritems():
            if vt == SINK_ROAD:
                logVhatsumVec[cvtInd] = -np.inf
                continue

            cThetahat = postHat.Theta[vt]
            if cThetahat.numComps() > 0:
                clogVhatsum = postHat.Theta[vt].normalizeWeights()
            else:
                clogVhatsum = -np.inf

            if np.isfinite(clogVhatsum):
                postHat.logV[vt] = clogVhatsum
                logVhatsumVec[cvtInd] = clogVhatsum
            else:
                logVhatsumVec[cvtInd] = -np.inf
                postHat.logV[vt] = -np.inf
                postHat.Theta[vt] = ()
        logVhatsum = logsumexp(logVhatsumVec)

        if normalizeLogV:
            # If we're looking at the whole distribution, normalize it.
            for vt in mapgraph.nodes_iter():
                if vt == SINK_ROAD:
                    continue
                postHat.logV[vt] -= logVhatsum
                logDiscardCompThresh = -50.0 + (np.log(mapgraph.node[vt]['length']) - np.log(mapgraph.totalLength))
                if np.isfinite(postHat.logV[vt]) and postHat.logV[vt] <= logDiscardCompThresh:
                    postHat.logV[vt] = -np.inf
                    postHat.Theta[vt] = ()
                    if debugFilter:
                        print '  logV = {1} < {2}, dropping street'.format(vt,postHat.logV[vt],logDiscardCompThresh)

        # Draw posterior: visualize all the means of the mixture models
        if drawPosterior:
            fig = plt.figure(figsize=displayFigSz)
            ax = fig.add_subplot(111)
            ax.scatter(gt_pos[0], gt_pos[1], s=0.2, c='r', label='gt')
            intersection_offset = 0.2  # 0.2 / project_scale * 1000 (m)

            for vt in allStreets:
                if vt == SINK_ROAD:
                    continue
                if vt in new_transition_streets:
                    postHat.logV[vt] = np.log(1.0 / len(new_transition_streets))
                if not np.isfinite(postHat.logV[vt]):
                    continue

                ThetaDist = postHat.Theta[vt]
                LogVstreet = postHat.logV[vt]

                # logWeightSumStreet = np.log(np.sum(np.exp(ThetaDist.getLogWs())))
                # print 'posterior vt = {0} heading = {1}'.format(vt, mapgraph.get_road_heading(vt, [0.0]))
                # print 'origin x = {0}, origin y = {1}, terminus x = {2}, terminus y = {3}'.format(\
                #     mapgraph.node[vt]['origin'][0], mapgraph.node[vt]['origin'][1],\
                #     mapgraph.node[vt]['terminus'][0], mapgraph.node[vt]['terminus'][1])

                for i in range(ThetaDist.numComps()):
                    if i > 1:
                        break
                    logw = ThetaDist.getLogW(i)
                    mu = ThetaDist.getMu(i)
                    # print 'mu = {0}'.format(mu)
                    sigma = ThetaDist.getSigma(i)
                    theta = mu.T
                    normalized_w = np.exp(logw + LogVstreet)
                    post_thetas = self.gaussian_sampling(True, mu, sigma, 100)
                    posterior_poses = trans_pose_global(mapgraph, vt, post_thetas)
                    ax.scatter(posterior_poses[:, 0], posterior_poses[:, 1], s=0.2, c='black', alpha=0.1)
                    curr_pos = trans_pose_global( mapgraph, vt, theta)[0]
                    lines = []
                    ax.plot([mapgraph.node[vt]['origin'][0], mapgraph.node[vt]['terminus'][0]], \
                            [mapgraph.node[vt]['origin'][1], mapgraph.node[vt]['terminus'][1]], \
                            linewidth=0.3, c='black')
                    street_xlim = mapgraph.node[vt]['xlim']
                    street_ylim = mapgraph.node[vt]['ylim']
                    intersect_segments_idx = np.array(
                        list(mapgraph.rtreeIndex.intersection((street_xlim[0] - intersection_offset, \
                                                               street_ylim[0] - intersection_offset, \
                                                               street_xlim[1] + intersection_offset, \
                                                               street_ylim[1] + intersection_offset))))
                    for l_idx in range(intersect_segments_idx.shape[0]):
                        line = all_segments[intersect_segments_idx[l_idx]]
                        lines.append(np.vstack((line[0:2], line[2:4])))
                    ax.add_collection(
                        LineCollection(lines, transOffset=ax.transData, linewidths=0.1, colors='green'))
                    ax.scatter(curr_pos[0], curr_pos[1], s=0.2, c='b', alpha=normalized_w)
                    ax.quiver(curr_pos[0], curr_pos[1], np.cos(curr_pos[2]), np.sin(curr_pos[2]), \
                              width=0.002, color='b', alpha=normalized_w)
                    ax.scatter(mapgraph.node[vt]['origin'][0], mapgraph.node[vt]['origin'][1], s=0.4, c='g')
                # print ''
            ax.legend()
            ax.set_title('frame idx: ' + str(frame_idx))
            ax.set_xlim([gt_pos[0] - intersection_offset, gt_pos[0] + intersection_offset])
            ax.set_ylim([gt_pos[1] - intersection_offset, gt_pos[1] + intersection_offset])
            directory = cwd + '/results/visualization/' + options.dataname + '/'
            fig_name = directory + str(frame_idx).zfill(6) + '_' 'posterior.png'
            fig.savefig(fig_name, dpi=displayDPI)
            plt.close('all')

        # self.draw_sample(frame_idx, gt_pos, local_normals, all_segments)
        

        if debugFilter:
            print 'End of radarFilter3.'
            print 'Radar filtering done in {0}s.'.format(time.time() - tmpt)

        return (postHat,logVhatsum)

