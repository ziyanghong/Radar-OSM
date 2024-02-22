import cython
import cProfile, pstats, StringIO
import numpy as np
# cython: profile=True

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
np.import_array()

from cython.parallel import prange 
from cython.parallel cimport parallel
cimport openmp
# cdef int num_threads

# openmp.omp_set_dynamic(1)
# with nogil, parallel():
#     num_threads = openmp.omp_get_num_threads()
#     print 'Number of thread'
#     print(num_threads)

###---------------------------------------------------adding types to variable-----------------------------------------------------------###
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime type info object.
DTYPE = np.double
ctypedef np.double_t DTYPE_t
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def projection_point_and_distances(np.ndarray[DTYPE_t, ndim=2] local_points_x_y_1, np.ndarray[DTYPE_t, ndim=3] poses_SE2, \
    np.ndarray[DTYPE_t, ndim=3] project_mat, np.int_t num_intersect_segments ):
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
    # prof = cProfile.Profile()
    # prof.enable()
    # Declare typed variables
    cdef int num_pose = poses_SE2.shape[2]
    cdef int num_point = local_points_x_y_1.shape[1]
    cdef int p, l, d, i, j
    cdef double m_square, inv_1_plus_m_square
    cdef np.ndarray[DTYPE_t, ndim=3] distance_mat = np.ones((num_pose, num_intersect_segments, num_point),  dtype=DTYPE) 
    cdef np.ndarray[DTYPE_t, ndim=3] projections = np.zeros((num_pose, num_intersect_segments, 2*num_point),  dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] m_square_array = project_mat[1,0,:]**2

    # cdef np.ndarray[DTYPE_t, ndim=2] global_points_x0_y0_1 = np.ones((3, num_point),  dtype=DTYPE)
    # global_points_x0_y0_1 = np.ones((3, num_point),  dtype=DTYPE)
    # cdef double [:,:] global_points_x0_y0_1_view = global_points_x0_y0_1

    cdef np.ndarray[DTYPE_t, ndim=2] projection_points =  np.zeros((2, num_point),  dtype=DTYPE)
    # cdef np.ndarray[DTYPE_t, ndim=1] x0_minus_xp, y0_minus_yp
    cdef double x0_minus_xp, y0_minus_yp
    cdef double projection_point_x, projection_point_y
    cdef double global_point_x0, global_point_y0

    # cdef double [:, :, :] distance_view = distance_mat


    # for p in range(num_pose):
        # global_points_x0_y0_1 = np.dot(poses_SE2[:,:,p], local_points_x_y_1)
    for l in prange(num_intersect_segments,  nogil=True, num_threads=16):
        m_square = m_square_array[l]
        inv_1_plus_m_square = 1.0 / (1.0 + m_square)
        for p in range(num_pose):
            for i in range(num_point):


                global_point_x0 = poses_SE2[0,0,p] * local_points_x_y_1[0,i] + poses_SE2[0,1,p] * local_points_x_y_1[1,i] + poses_SE2[0,2,p]
                global_point_y0 = poses_SE2[1,0,p] * local_points_x_y_1[0,i] + poses_SE2[1,1,p] * local_points_x_y_1[1,i] + poses_SE2[1,2,p]

                projection_point_x = (project_mat[0,0,l] * global_point_x0 + project_mat[0,1,l] * global_point_y0 + project_mat[0,2,l]) * inv_1_plus_m_square
                projection_point_y = (project_mat[1,0,l] * global_point_x0 + project_mat[1,1,l] * global_point_y0 + project_mat[1,2,l]) * inv_1_plus_m_square
                projections[p, l, i] = projection_point_x
                projections[p, l, i + num_point] = projection_point_y
                x0_minus_xp = projection_point_x - global_point_x0
                y0_minus_yp = projection_point_y - global_point_y0
                distance_mat[p,l,i] = x0_minus_xp**2 + y0_minus_yp**2

    distance_mat = np.sqrt(distance_mat)
    return projections, distance_mat


# ###---------------------------------------------------adding types to variable-----------------------------------------------------------###
# # We now need to fix a datatype for our arrays. I've used the variable
# # DTYPE for this, which is assigned to the usual NumPy runtime type info object.
# DTYPE = np.double
# ctypedef np.double_t DTYPE_t
# cimport cython
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# def projection_point_and_distances(np.ndarray[DTYPE_t, ndim=2] local_radar_points, np.ndarray[DTYPE_t, ndim=3] poses_SE2, \
#     np.ndarray[DTYPE_t, ndim=3] project_mat, np.int_t num_intersect_segments):
#     """
#     Input:
#         local_radar_points: 4 x N
#         poses_SE2: 3 x 3 x P
#         project_mat:  2 x 3 x L
#         num_intersect_segments: L
#     Output:
#         projections: P x L x 2N
#         distance_mat: P x L x N
#     """
#     # prof = cProfile.Profile()
#     # prof.enable()
#     # Declare typed variables
#     cdef int num_pose = poses_SE2.shape[2]
#     cdef int num_point = local_radar_points.shape[1]
#     cdef int p, l, d, i, j
#     cdef double m_square
#     cdef np.ndarray[DTYPE_t, ndim=2] local_points_x_y_1 = local_radar_points
#     cdef np.ndarray[DTYPE_t, ndim=3] distance_mat = np.zeros((num_pose, num_intersect_segments, num_point),  dtype=DTYPE)
#     cdef np.ndarray[DTYPE_t, ndim=3] projections = np.zeros((num_pose, num_intersect_segments, 2*num_point),  dtype=DTYPE)
#     cdef np.ndarray[DTYPE_t, ndim=2] global_points_x0_y0_1 = np.ones((3, num_point),  dtype=DTYPE)
#     # global_points_x0_y0_1 = np.ones((3, num_point),  dtype=DTYPE)
#     # cdef double [:,:] global_points_x0_y0_1_view = global_points_x0_y0_1

#     cdef np.ndarray[DTYPE_t, ndim=2] projection_points =  np.zeros((2, num_point),  dtype=DTYPE)
#     # cdef np.ndarray[DTYPE_t, ndim=1] x0_minus_xp, y0_minus_yp
#     cdef double x0_minus_xp, y0_minus_yp

#     cdef double [:, :, :] distance_view = distance_mat
#     # cdef double [:, :, :] projections_view = projections

#     # cdef double [:,:] local_points_view = local_points_x_y_1
#     # cdef double [:,:,:] pose_view = poses_SE2

#     # for p in prange(num_pose, nogil=True):
#     for p in range(num_pose):
#         global_points_x0_y0_1 = np.dot(poses_SE2[:,:,p], local_points_x_y_1)
#         # global_points_x0_y0_1 = transform_global_pose(poses_SE2[:,:,p], local_points_x_y_1)

#         # for j in range(num_point):
#         #     global_points_x0_y0_1_view[0, j] = pose_view[0,0,p] * local_points_view[0,j] + pose_view[0,1,p] * local_points_view[1,j] + pose_view[0,2,p]
#         #     global_points_x0_y0_1_view[1, j] = pose_view[1,0,p] * local_points_view[0,j] + pose_view[1,1,p] * local_points_view[1,j] + pose_view[1,2,p]

#         # for l in prange(num_intersect_segments, nogil=True):
#         for l in range(num_intersect_segments):
#             m_square = project_mat[1,0,l]**2
#             projection_points = np.dot(project_mat[:,:,l], global_points_x0_y0_1) / (1 + m_square)
#             # projections[p, l, 0:num_point] = projection_points[0,:]
#             # projections[p, l, num_point:2*num_point] = projection_points[1,:]
#             # x0_minus_xp = np.add(projection_points[0, :], -global_points_x0_y0_1[0, :])
#             # y0_minus_yp = np.add(projection_points[1, :], -global_points_x0_y0_1[1, :])
#             # distance_mat[p,l,:] = np.sqrt(x0_minus_xp**2 + y0_minus_yp**2)
#             for i in range(num_point):
#                 projections[p, l, i] = projection_points[0,i]
#                 projections[p, l, i + num_point] = projection_points[1,i]
#                 x0_minus_xp = projection_points[0, i] - global_points_x0_y0_1[0, i]
#                 y0_minus_yp = projection_points[1, i] - global_points_x0_y0_1[1, i]
#                 distance_view[p,l,i] = x0_minus_xp**2 + y0_minus_yp**2


#     distance_mat = np.sqrt(distance_mat)
#     # prof.disable()
#     # stringIO = StringIO.StringIO()
#     # sortby = 'cumulative'
#     # ps = pstats.Stats(prof, stream=stringIO).sort_stats(sortby)
#     # ps.print_stats()
#     # print stringIO.getvalue()
#     return projections, distance_mat



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


# def projection_point_and_distances(local_radar_points, poses_SE2, project_mat, num_intersect_segments):
#     """
#     Input:
#         local_radar_points: 4 x N
#         poses_SE2: 3 x 3 x P
#         project_mat:  2 x 3 x L
#         num_intersect_segments: L
#     Output:
#         projections: P x L x 2N
#         distance_mat: P x L x N
#     """

#     # Declare typed variables
#     cdef int num_pose = poses_SE2.shape[2]
#     cdef int num_point = local_radar_points.shape[1]
#     cdef Py_ssize_t p, l, d, i
#     distance_mat = np.zeros((num_pose, num_intersect_segments, num_point),  dtype=np.double)
#     projections = np.zeros((num_pose, num_intersect_segments, 2*num_point),  dtype=np.double)
#     # cdef double [:, :, :] distance_view = distance_mat
#     # cdef double [:, :, :] projections_view = projections


#     local_points_x_y_1 = local_radar_points[0:3, :]
#     # for p in prange(num_pose, nogil=True):
#     for p in range(num_pose):
#         global_points_x0_y0_1 = np.dot(poses_SE2[:,:,p], local_points_x_y_1)
#         # for l in prange(num_intersect_segments, nogil=True):
#         for l in range(num_intersect_segments):
#             m_square = project_mat[1,0,l]**2
#             projection_points = np.dot(project_mat[:,:,l], global_points_x0_y0_1) / (1 + m_square)
#             # global_points_x0_minus =  -global_points_x0_y0_1[0, :]
#             # global_points_y0_minus =  -global_points_x0_y0_1[1, :]
#             x0_minus_xp = np.add(projection_points[0, :], -global_points_x0_y0_1[0, :])
#             y0_minus_yp = np.add(projection_points[1, :], -global_points_x0_y0_1[1, :])
#             distances = np.sqrt(x0_minus_xp**2 + y0_minus_yp**2)
#             distance_mat[p,l,:] = distances
#             projections[p, l, :] = projection_points.flatten('F')

#             # distance_view[p,l,:] = distances
#             # projections_view[p, l, :] = projection_points.flatten('F')


#     return projections, distance_mat


