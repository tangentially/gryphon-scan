# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import numpy as np
from scipy import optimize
import cv2

from horus import Singleton
from horus.engine.calibration.calibration import CalibrationCancel
from horus.engine.calibration.moving_calibration import MovingCalibration
from horus.gui.util.augmented_view import augmented_draw_pattern

import logging
logger = logging.getLogger(__name__)


class PlatformExtrinsicsError(Exception):

    def __init__(self):
        Exception.__init__(self, "PlatformExtrinsicsError")


#estimated_t = [-5, 90, 320]

class MarkerData(object):

    def __init__(self, name, h = None):
        self.name = name
        self.h = h

        self.x = []
        self.y = []
        self.z = []

        # circle
        self.center = None
        self.circle = None
        self.radius = None

        # platform
        self.R = None
        self.t = None
        self.n = None

    def is_calibrated(self):
        return self.R is not None and \
               self.n is not None

    def put(self, x, y, z):
        self.x += [x]
        self.y += [y]
        self.z += [z]

    def calibrate(self):
        self.R = None
        self.t = None
        self.n = None

        self.center = None
        self.circle = None
        self.radius = None

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)
        points = zip(self.x, self.y, self.z)

        if len(points) > 4:
            # Fitting a plane
            point, normal = fit_plane(points)
            if normal[1] > 0:
                normal = -normal
            self.n = normal
            # Fitting a circle inside the plane
            self.center, self.R, self.circle, self.radius = fit_circle(point, normal, points)
            # Get platform origin (if known => can find platform center)
            if self.h is not None:
                self.t = self.center - self.h * np.array(normal)

            logger.info("Marker calibration (" + self.name + ')')
            logger.info(" Translation: " + str(self.t))
            #logger.info(" Rotation: " + str(self.R).replace('\n', ''))
            logger.info(" Normal: " + str(self.n))

            return True
        return False


class NormalData(MarkerData):
    def calibrate(self):
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)
        points = zip(self.x, self.y, self.z)

        if len(points) > 3:
            # Fitting rotation axis for normals
            normal = fit_rot_vec(points)
            if normal[1] > 0:
                normal = -normal
            self.n = normal
            self.R = make_R(self.n)


@Singleton
class PlatformExtrinsics(MovingCalibration):

    def __init__(self):
        self.image = None
        self.has_image = False
        self.points_image = None
        MovingCalibration.__init__(self)
        self.use_chessboard = False
        self.use_aruco = False

    def _initialize(self):
        self.image = None
        self.has_image = True
        self.image_capture.stream = False

        self.data = { 'c0': MarkerData('Chessboard pose: platform',
                                0), 
                      'c1': MarkerData('Chessboard pose: origin', 
                                self.pattern.square_width * (self.pattern.rows-1) + self.pattern.origin_distance), 
                      'c2': MarkerData('Chessboard projection: distorted',
                                self.pattern.origin_distance), # from bottom corner projection distorted
                      'c3': MarkerData('Chessboard projection: undistorted',
                                self.pattern.origin_distance), # from bottom corner projection distorted
                      'n':  NormalData('Chessboard normals', None),
                    } 

        # detect which markers to use
        self.use_chessboard = False
        self.use_aruco = False
        image = self.image_capture.capture_pattern()
        corners = self.image_detection.detect_corners(image)
        if corners is not None:
            self.use_chessboard = True
        corners, ids = self.image_detection.aruco_detect(image)
        if corners:
            self.use_aruco = True
        if not self.use_chessboard and not self.use_aruco:
            self._is_calibrating = False

    def _capture(self, angle):
        image = self.image_capture.capture_pattern()

        if self.points_image is None:
            self.points_image = np.zeros(image.shape, dtype = "uint8")
        # ============ chessboard pattern ===========
        pose = None
        if self.use_chessboard:
            corners = self.image_detection.detect_corners(image, False)
            pose = self.image_detection.detect_pose_from_corners(corners)
        if pose is not None:
            #image = self.image_detection.draw_pattern(image, corners)
            # TODO: Move all visualizaton AFTER detection
            image = augmented_draw_pattern(image, corners)

            print("\n---- platform_extrinsics --- "+str(angle))

            # ----- Points from pattern pose -----
            # detect_pose() uses distortion while estimate pattern pose
            rvec = pose[0]
            tvec = pose[1].T[0]

            # -- normal
            n = rvec.T[0]
            self.data['n'].put(n[0], n[1], n[2])
            
            # -- Top point
            self.data['c1'].put(tvec[0], tvec[1], tvec[2])

            # -- Bottom point
            #point = np.float32([0, self.pattern.square_width * (self.pattern.rows-1) + self.pattern.origin_distance,0])
            #pp = rvec.dot(point) + tvec
            # optimized
            pp = (self.pattern.square_width * (self.pattern.rows-1) + self.pattern.origin_distance) * rvec.T[1] + tvec
            print(pp) # bottom point from pattern pose
            self.data['c0'].put(pp[0], pp[1], pp[2])

            #points = np.float32([p1, p2])
            p, jac = cv2.projectPoints([pp, tvec],
                np.identity(3),
                np.zeros(3),
                self.calibration_data.camera_matrix,
                self.calibration_data.distortion_vector)
            p = np.int32(p).reshape(-1,2)
            for pp in p:
              cv2.circle(self.points_image, tuple(pp), 5, (255,255,0), -1)

            '''
            # DEBUG: project bottom point to image coordinates
            p, jac = cv2.projectPoints(np.float32( [tuple(pp)] ), \
                np.identity(3),
                np.zeros(3),
                self.calibration_data.camera_matrix, \
                self.calibration_data.distortion_vector )
            print(p) # bottom point projection
            
            # ----- reconstruct bottom point from corner projection -----
            plane = self.image_detection.detect_pattern_plane(pose)
            if plane is not None:
                distance, normal, corners = plane
                #self.image = self.image_detection.draw_pattern(image, corners)
                if corners is not None:
                    print("--- from corners ---")
                    origin = corners[self.pattern.columns * (self.pattern.rows - 1)][0]
                    origin = np.array([[origin[0], corners[0][0][0]], [origin[1], corners[0][0][1]]])
                    #origin = p[0].T # debug: test with point projection from prev step
                    print(origin.T)

                    print("   - distorted -")
                    t = self.point_cloud_generation.compute_camera_point_cloud(
                        origin, distance, normal)
                    if t is not None:
                        self.data['c2'].put(t[0][0], t[1][0], t[2][0])
                        print( [t[0][0], t[1][0], t[2][0]])
                        print( np.array([t[0][0], t[1][0], t[2][0]]) - pp)
            
                    # ----- using undistort -----
                    print("   - undistorted -")
                    o = self.point_cloud_generation.undistort_points(origin)
                    t = self.point_cloud_generation.compute_camera_point_cloud(
                        o, distance, normal)
                    if t is not None:
                        self.data['c3'].put(t[0][0], t[1][0], t[2][0])
                        print( [t[0][0], t[1][0], t[2][0]])
                        print( np.array([t[0][0], t[1][0], t[2][0]]) - pp)
            '''

        # ============ ARUCO markers ===========
        corners = None
        if self.use_aruco:
            corners, ids = self.image_detection.aruco_detect(image)
        if corners is not None:
            image, rvecs, tvecs = self.image_detection.aruco_draw_markers(image, corners, ids)
            #print(rvecs.shape)
            tvecs = np.squeeze(tvecs, axis=1)
            print(tvecs.shape)
            #print(tvecs)
            for i, id in enumerate(ids):
               if 'ar'+str(id) not in self.data:
                   self.data.update({ 'ar'+str(id) : MarkerData('ARUCO #'+str(id), 20) })
               self.data['ar'+str(id)].put(tvecs[i][0], tvecs[i][1], tvecs[i][2], )

            #points = np.float32([p1, p2])
            p, jac = cv2.projectPoints(tvecs,
                np.identity(3),
                np.zeros(3),
                self.calibration_data.camera_matrix,
                self.calibration_data.distortion_vector)
            p = np.int32(p).reshape(-1,2)
            for pp in p:
              cv2.circle(self.points_image, tuple(pp), 5, (255,255,0), -1)

        # display image
        self.image = image
        np.maximum(self.image, self.points_image, out = self.image)

    def _calibrate(self):
        self.has_image = False
        self.image_capture.stream = True

        normal_avg = np.zeros(3)
        t_avg = np.zeros(3)
        t_avg_n = 0

        # calibrate each data set and calculate average results
        for i,d in self.data.iteritems():
            d.calibrate()
            if d.n is not None:
                normal_avg += d.n
            if d.t is not None:
                t_avg += d.t
                t_avg_n += 1

        normal_avg /= np.linalg.norm(normal_avg)
        R_avg = make_R(normal_avg)
        if t_avg_n>0:
            t_avg /= t_avg_n
        logger.info(" --- AVG all --- ")
        logger.info(" Normal: " + str( normal_avg ))
        logger.info(" Translation: " + str(t_avg))

        if self.data['c0'].is_calibrated() and \
           self.data['c1'].is_calibrated():

            # normal by centers
            normal_c = self.data['c1'].center - self.data['c0'].center
            normal_c /= np.linalg.norm(normal_c)
            R_c = make_R(normal_c)

            logger.info(" --- by Centers --- ")
            logger.info(" Normal: " + str( normal_c ))

            # average normals c0 c1
            normal_avg = self.data['c0'].n + self.data['c1'].n
            normal_avg /= np.linalg.norm(normal_avg)
            normal_avg = np.float32(normal_avg)
            t_avg = (self.data['c0'].t + self.data['c1'].t)/2
            R_avg = make_R(normal_avg)

            logger.info(" --- AVG c0 c1 --- ")
            logger.info(" Normal: " + str( normal_avg ))
            logger.info(" Translation: " + str(t_avg))

            # estimate rotation
            rsteps = 3
            data = zip( zip(self.data['c0'].x, self.data['c0'].y, self.data['c0'].z),
                        zip(self.data['c1'].x, self.data['c1'].y, self.data['c1'].z) )
            data1 = np.array(data[:-rsteps]).reshape(-1,3)
            data2 = np.array(data[rsteps:]).reshape(-1,3)
            R, t, centroid_A, centroid_B = rigid_transform_3D(data1, data2)
            
            Rv,_ = cv2.Rodrigues(R)
            l = np.linalg.norm(Rv)
            Rv = Rv.flatten()/l
            l = l*180/np.pi
            print("Rv: %s alpha: %s" % (str(Rv), str(l/rsteps)) )
            tr = PointOntoLine(t, Rv, centroid_A)

            logger.info(" --- Rotation SVD c0 c1 --- ")
            logger.info(" Normal: " + str( Rv ))
            logger.info(" Translation: " + str(tr))

        logger.info(" --- Rotation of pattern normal --- ")
        logger.info(" Normal: " + str( self.data['n'].n ))

        #normal_avg = np.array([0, -1, 0], dtype=np.float32)
        #R_avg = make_R(normal_avg)
        #t_avg = np.array([0, 106, 463], dtype=np.float32)

        if self._is_calibrating:
            if t_avg_n>0:
                self.n = normal_avg
                self.R = R_avg
                self.t = t_avg
                response = (True, (self.R, self.t, self.data))

            else:
                response = (False, PlatformExtrinsicsError())
        else:
            response = (False, CalibrationCancel())

        self._is_calibrating = False
        self.image = None

        return response

    def accept(self):
        self.calibration_data.platform_rotation = self.R
        self.calibration_data.platform_translation = self.t

    def set_estimated_size(self, estimated_size):
        global estimated_t
        estimated_t = estimated_size


def distance2plane(p0, n0, p):
    return np.dot(np.array(n0), np.array(p) - np.array(p0))


def residuals_plane(parameters, data_point):
    px, py, pz, theta, phi = parameters
    nx, ny, nz = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    distances = [distance2plane(
        [px, py, pz], [nx, ny, nz], [x, y, z]) for x, y, z in data_point]
    return distances


def fit_plane(data):
    estimate = [0, 0, 0, 0, 0]  # px,py,pz and zeta, phi
    # you may automize this by using the center of mass data
    # note that the normal vector is given in polar coordinates
    best_fit_values, ier = optimize.leastsq(residuals_plane, estimate, args=(data))
    xF, yF, zF, tF, pF = best_fit_values

    # point  = [xF,yF,zF]
    point = data[0]
    normal = -np.array([np.sin(tF) * np.cos(pF), np.sin(tF) * np.sin(pF), np.cos(tF)])

    return point, normal


def residuals_circle(parameters, points, s, r, point):
    r_, s_, Ri = parameters
    plane_point = s_ * s + r_ * r + np.array(point)
    distance = [np.linalg.norm(plane_point - np.array([x, y, z])) for x, y, z in points]
    res = [(Ri - dist) for dist in distance]
    return res


def make_R(normal):
    # creating two inplane vectors
    # assuming that normal not parallel x!
    s = np.cross(np.array([1, 0, 0]), np.array(normal))
    s = s / np.linalg.norm(s)
    r = np.cross(np.array(normal), s)
    r = r / np.linalg.norm(r)  # should be normalized already, but anyhow

    # Define rotation
    return np.array([s, r, normal]).T


def fit_circle(point, normal, points):
    # creating two inplane vectors
    # assuming that normal not parallel x!
    s = np.cross(np.array([1, 0, 0]), np.array(normal))
    s = s / np.linalg.norm(s)
    r = np.cross(np.array(normal), s)
    r = r / np.linalg.norm(r)  # should be normalized already, but anyhow

    # Define rotation
    R = np.array([s, r, normal]).T

    estimate_circle = [0, 0, 0]  # px,py,pz and zeta, phi
    best_circle_fit_values, ier = optimize.leastsq(
        residuals_circle, estimate_circle, args=(points, s, r, point))

    rF, sF, RiF = best_circle_fit_values

    # Synthetic Data
    center_point = sF * s + rF * r + np.array(point)
    synthetic = [list(center_point + RiF * np.cos(phi) * r + RiF * np.sin(phi) * s)
                 for phi in np.linspace(0, 2 * np.pi, 50)]
    [cxTupel, cyTupel, czTupel] = [x for x in zip(*synthetic)]

    return center_point, R, [cxTupel, cyTupel, czTupel], RiF


# --------- estimate rotation axis for set of rotated vectors ----------
def residuals_rot_vec(parameters, data_vectors):
    theta, phi = parameters
    v = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    distances =  [np.dot(v, [x, y, z]) for x, y, z in data_vectors]
    return distances


def fit_rot_vec(data):
    estimate = [-np.pi/2, np.pi/2]  # theta, phi
    best_fit_values, ier = optimize.leastsq(residuals_rot_vec, estimate, args=(data))
    tF, pF = best_fit_values

    print("Fit vec: Theta %s Phi %s" % (str(tF), str(pF)))
    v = np.array([np.sin(tF) * np.cos(pF), np.sin(tF) * np.sin(pF), np.cos(tF)])

    return v


# Finding optimal rotation and translation between corresponding 3D points
# http://nghiaho.com/?page_id=671
#
# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    print(A.shape)
    print(B.shape)
    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    print(centroid_A)
    print(centroid_B)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.matmul(np.transpose(AA), BB)

    U, S, Vt = np.linalg.svd(H)

    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = np.matmul(Vt.T, U.T)

    t = -np.matmul(R,centroid_A.T) + centroid_B.T

    print t

    return R, t, centroid_A, centroid_B

# Point to line projection
#  a, v - line point and vector 
#  p - point to project
def PointOntoLine(a, v, p):
    ap = p-a
    result = a + np.dot(ap,v)/np.dot(v,v) * v
    return result

