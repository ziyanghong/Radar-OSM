import numpy as np
import math
from numba import njit, jit
from numba.numpy_extensions import cross2d
import os
cwd = os.getcwd()
displayFigSz = (7,7)


def wrap_angle(angle):
    """wrap angle to positive value from 0 to pi"""
    if angle < 0.0:

        wraped_angle = angle + np.pi
    else:
        wraped_angle = angle
    return wraped_angle


def SE2_to_se2(SE2):
    x = SE2[0,2]
    y = SE2[1,2]
    heading = np.arctan2(SE2[1, 0], SE2[0, 0])
    se2 = np.asarray([x, y, heading])
    return se2

def se2_to_SE2(se2):
    """
    Input:
        se2 = [x, y , angle] 3
    Output:

    """
    SE2 = np.array([[math.cos(se2[2]), -math.sin(se2[2]), se2[0]],
                    [math.sin(se2[2]),  math.cos(se2[2]), se2[1]],
                    [               0,                 0,     1]])
    return SE2

# @jit(nopython=True)
# @njit('float64(float64[:], float64[:],float64[:])')
@njit
def projection_distance(line_origin, line_terminus, point):

    # distance_projection = np.float64(0.0)
    cross_product = cross2d(line_origin - line_terminus,  line_origin - point)
    # print cross_product
    norm_numerator = np.abs(cross_product)
    norm_denominator =  np.linalg.norm(line_origin - line_terminus)
    distance_projection =  norm_numerator / norm_denominator

    return distance_projection

@jit
def compute_norm(a):
    sum = np.float64(0.0)
    # print a.shape[0]
    for i in range(a.shape[0]):
        sum += a[i]*a[i]
    norm = np.sqrt(sum)
    return norm

@jit
def compute_sum(a):
    sum = 0.0
    for i in range(a.shape[0]):
        sum += a[i]
    return sum



def inverse_pose(pose):
    """pose: SE2 Pose"""
    inv_pose = np.eye(3)
    Rcw = pose[0:2,0:2]
    tcw = pose[0:2,2]
    Rwc = Rcw.transpose()
    Ow  = np.dot(-Rwc, tcw)
    inv_pose[0:2,0:2] = Rwc
    inv_pose[0:2,2] = Ow
    return inv_pose


def trans_pose_global(mapgraph, street, thetas):
    """
    Transform the theta samples to global poses

    Input:
        thetas = [x, y, angle] N x 3 matrix, respect to local street coordinate
    Output:
        global_poses_se2 = [x, y, heading] N x 3 matrix

    """
    # print thetas
    global_poses_se2 = np.zeros((thetas.shape[0], 3))
    street_heading = mapgraph.get_road_heading(street, [0.0])
    street_origin = mapgraph.get_road_position(street, [0.0])
    street_origin_pose = se2_to_SE2(np.asarray([street_origin[0], street_origin[1], street_heading]))
    for i in range(thetas.shape[0]):
        local_pose_SE2 = se2_to_SE2(np.asarray([thetas[i, 0], thetas[i, 1], thetas[i, 2]]))
        global_pose_SE2 = np.dot(street_origin_pose, local_pose_SE2)
        global_poses_se2[i, 0] = global_pose_SE2[0, 2]
        global_poses_se2[i, 1] = global_pose_SE2[1, 2]
        global_poses_se2[i, 2] = np.arctan2(global_pose_SE2[1,0], global_pose_SE2[0,0])
    return global_poses_se2

# deprecated
def trans_pose_global_old(mapgraph, street, thetas):
    """
    Transform the theta samples to global poses

    Input:
        thetas = [distance_now, distance_previous, heading_now, heading_previous, offset] N x 5 matrix
    Output:
        global_poses = [x, y, heading] N x 3 matrix

    """
    global_poses = np.zeros((thetas.shape[0], 3))
    for i in range(thetas.shape[0]):
        road_heading = mapgraph.get_road_heading(street, [thetas[i, 0]])
        global_position = mapgraph.get_road_position(street, [thetas[i, 0]])

        if (thetas[i, 2] >= -np.pi / 2) and (thetas[i, 2] <= np.pi / 2):
            x = global_position[0] - thetas[i, 4] * np.sin(road_heading)
            y = global_position[1] + thetas[i, 4] * np.cos(road_heading)

        else:
            # print 'backward, drive right'
            x = global_position[0] - thetas[i, 4] * np.sin(road_heading)
            y = global_position[1] + thetas[i, 4] * np.cos(road_heading)

        heading = thetas[i, 2] + road_heading
        global_poses[i, 0] = x
        global_poses[i, 1] = y
        global_poses[i, 2] = heading
    return global_poses

def trans_points_global(points, global_heading, global_position):
    transformation = np.asarray([[np.cos(global_heading), -np.sin(global_heading), 0], \
                                 [np.sin(global_heading), np.cos(global_heading), 0], \
                                 [0, 0, 1]], dtype=points.dtype)
    transformed_points = np.dot(transformation, points[0:3, :])
    transformed_points[0, :] = transformed_points[0, :] + global_position[0]
    transformed_points[1, :] = transformed_points[1, :] + global_position[1]
    return transformed_points


def pose_to_dist_angle_offset(global_pose, street, mapgraph):
    street_origin = mapgraph.node[street]['origin']
    street_terminus = mapgraph.node[street]['terminus']

    # distance between p1 and p2
    l2 = np.sum((street_origin - street_terminus) ** 2)
    # if you need the point to project on line segment between p1 and p2 or closest point of the line segment
    t = max(0, min(1, np.sum((global_pose[0:2] - street_origin) * (street_terminus - street_origin)) / l2))
    projection = street_origin + t * (street_terminus - street_origin)
    distance2origin = np.sqrt((projection[0] - street_origin[0])**2 +\
                              (projection[1] - street_origin[1])**2)
    street_heading = mapgraph.get_road_heading(street, distance2origin)
    angle = global_pose[2] - street_heading
    offset = global_pose_to_local_offset(mapgraph, street, global_pose.reshape(3))

    return distance2origin, angle, offset

@njit
def trans_lines_local(lines, global_heading, global_position):
    lines_start = np.vstack((lines[:,0:2].transpose(), np.ones((1, lines.shape[0]))))
    lines_end = np.vstack((lines[:,2:4].transpose(), np.ones((1, lines.shape[0]))))
    c = np.cos(global_heading)
    s = np.sin(global_heading)
    x = global_position[0]
    y = global_position[1]
    pose = np.array([[c, -s, x],\
                     [s,  c, y],\
                     [0.0, 0.0, 1.0]])
    inv_pose = inverse_pose(pose)
    lines_start_local = np.dot(inv_pose, lines_start)
    lines_end_local = np.dot(inv_pose, lines_end)
    transformed_lines = np.hstack((lines_start_local[0:2,:].transpose(),\
                                   lines_end_local[0:2,:].transpose()))
    return transformed_lines

def global_pose_to_local_offset(mapgraph, street, global_pose):
    mercator_xy = global_pose[0:2]
    # mercator_heading = global_pose[2]
    street_origin = mapgraph.node[street]['origin']
    street_terminus = mapgraph.node[street]['terminus']
    cross_product = np.cross(street_origin-street_terminus,street_origin-mercator_xy)
    offset2street = np.linalg.norm(cross_product)/ \
                          np.linalg.norm(street_origin-street_terminus)
    distance2street = offset2street.reshape((1))
    street_vector = street_terminus - street_origin
    origin_2_offset_pt_vector = global_pose[0:2] - street_origin
    street_heading = np.arctan2(street_vector[1], street_vector[0])
    origin_2_offset_pt_vector_headng = np.arctan2(origin_2_offset_pt_vector[1], origin_2_offset_pt_vector[0])
    if origin_2_offset_pt_vector_headng > street_heading :
        offset2street = distance2street
    else:
        offset2street = -distance2street
    return offset2street



def global_pose_to_local_pose( mapgraph, street, global_pose_obst, options, frame_id, gt_pos):
    """
    Transform global pose observation to local pose observation
    Input:
        global_pose_obst = [x, y, heading]
    Output:
        local_pose_obst = [x, y, heading] respect to street origin

    """
    local_pose_obst = np.zeros((1, 3))
    st_origin = mapgraph.node[street]['origin']
    st_heading = mapgraph.get_road_heading(street, [0.0])
    street_origin_pose_SE2 = se2_to_SE2(np.asarray([st_origin[0], st_origin[1], st_heading]))
    global_pose_obst_SE2 = se2_to_SE2(global_pose_obst)
    local_pose_obst_SE2 = np.dot(inverse_pose(street_origin_pose_SE2), global_pose_obst_SE2)
    local_pose_obst_se2 = SE2_to_se2(local_pose_obst_SE2)
    local_pose_obst[0,:] = local_pose_obst_se2


    return local_pose_obst
