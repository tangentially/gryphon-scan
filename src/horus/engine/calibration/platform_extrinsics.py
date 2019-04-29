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


@Singleton
class PlatformExtrinsics(MovingCalibration):

    """Platform extrinsics algorithm:

            - Rotation matrix
            - Translation vector
    """

    def __init__(self):
        self.image = None
        self.has_image = False
        MovingCalibration.__init__(self)

    def _initialize(self):
        self.image = None
        self.has_image = True
        self.image_capture.stream = False
        self.x = []
        self.y = []
        self.z = []

        self.x1 = []
        self.y1 = []
        self.z1 = []

    def _capture(self, angle):
        image = self.image_capture.capture_pattern()
        corners = self.image_detection.detect_corners(image)
        pose = self.image_detection.detect_pose_from_corners(corners)
#        pose = self.image_detection.detect_pose(image)
        if pose is not None:
            print("\n---- platform_extrinsics ---")
            # detect_pose() uses distortion while estimate pattern pose

            # ----- bottom point from pattern pose -----
            #rvec = pose[0]
            tvec = pose[1].T[0]
            #point = np.float32([0, self.pattern.square_width * (self.pattern.rows-1),0])
            #pp = rvec.dot(point) + tvec
            pp = (self.pattern.square_width * (self.pattern.rows-1)) * pose[0].T[1] + pose[1].T[0]
            #pp = (self.pattern.square_width * (self.pattern.rows-1) + self.pattern.origin_distance) * pose[0].T[1] + pose[1].T[0]
            print(pp) # bottom point from pattern pose
            self.x += [pp[0]]
            self.y += [pp[1]]
            self.z += [pp[2]]
            #self.image = self.image_detection.draw_pattern(image, corners)
            self.image = augmented_draw_pattern(image, corners)

            self.x1 += [tvec[0]]
            self.y1 += [tvec[1]]
            self.z1 += [tvec[2]]

            p, jac = cv2.projectPoints(np.float32( [tuple(pp)] ), \
                np.identity(3),
                np.zeros(3),
                self.calibration_data.camera_matrix, \
                self.calibration_data.distortion_vector )
            print(p) # bottom point projection
            
            # ----- bottom point from corner projection -----
            plane = self.image_detection.detect_pattern_plane(pose)
            if plane is not None:
                distance, normal, corners = plane
                #self.image = self.image_detection.draw_pattern(image, corners)
                if corners is not None:
                    # ----- original Ciclop -----
                    print("--- distorted ---")
                    origin = corners[self.pattern.columns * (self.pattern.rows - 1)][0]
                    origin = np.array([[origin[0], corners[0][0][0]], [origin[1], corners[0][0][1]]])
                    #origin = p[0].T # debug: test with point projection from prev step
                    print(origin.T)

                    t = self.point_cloud_generation.compute_camera_point_cloud(
                        origin, distance, normal)
                    if t is not None:
                        #self.x += [t[0][0]]
                        #self.y += [t[1][0]]
                        #self.z += [t[2][0]]
                        print( [t[0][0], t[1][0], t[2][0]])
                        print( np.array([t[0][0], t[1][0], t[2][0]]) - pp)
            
                    # ----- using undistort -----
                    o = self.point_cloud_generation.undistort_points(origin)
                    t = self.point_cloud_generation.compute_camera_point_cloud(
                        o, distance, normal)
                    if t is not None:
                        print( [t[0][0], t[1][0], t[2][0]])
                        print( np.array([t[0][0], t[1][0], t[2][0]]) - pp)
            
        else:
            self.image = image

    def _calibrate(self):
        self.has_image = False
        self.image_capture.stream = True
        self.t = None
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)
        points = zip(self.x, self.y, self.z)

        self.x1 = np.array(self.x1)
        self.y1 = np.array(self.y1)
        self.z1 = np.array(self.z1)
        points1 = zip(self.x1, self.y1, self.z1)

        if len(points) > 4:
            # Fitting a plane
            point, normal = fit_plane(points)
            if normal[1] > 0:
                normal = -normal
            # Fitting a circle inside the plane
            center, self.R, circle = fit_circle(point, normal, points)
            # Get real origin
            self.t = center - self.pattern.origin_distance * np.array(normal)
            #self.t = center

            logger.info("Platform calibration ")
            logger.info(" Translation: " + str(self.t))
            logger.info(" Rotation: " + str(self.R).replace('\n', ''))
            logger.info(" Normal: " + str(normal))

            # ==== top circle points
            point1, normal1 = fit_plane(points1)
            if normal1[1] > 0:
                normal1 = -normal1
            # Fitting a circle inside the plane
            center1, self.R1, circle1 = fit_circle(point1, normal1, points1)
            # Get real origin
            self.t1 = center1 - (self.pattern.square_width * (self.pattern.rows-1) + self.pattern.origin_distance) * np.array(normal)

            normal_c = center1 - center
            normal_c /= np.linalg.norm(normal_c)
            R_c = make_R(normal_c)

            normal_avg = (normal + normal1)/2
            center_avg = (self.t + self.t1)/2
            R_avg = make_R(normal_avg)

            logger.info(" --- TOP --- ")
            logger.info(" Translation Top: " + str(self.t1))
            logger.info(" Rotation Top: " + str(self.R1).replace('\n', ''))
            logger.info(" Normal Top: " + str(normal1))
            logger.info(" --- by Centers --- ")
            logger.info(" Normal by Centers: " + str( normal_c ))
            logger.info(" Rotation : " + str(R_c).replace('\n', ''))
            logger.info(" --- AVG --- ")
            logger.info(" Normal avg: " + str( normal_avg ))
            logger.info(" Rotation : " + str(R_avg).replace('\n', ''))

        if self._is_calibrating and self.t is not None: # and \
#           np.linalg.norm(self.t - estimated_t) < 100:
#            response = (True, (self.R, self.t, center, point, normal,
#                        [self.x, self.y, self.z], circle))
            response = (True, (self.R, center_avg, center, point, normal,
                        [self.x, self.y, self.z], circle))

        else:
            if self._is_calibrating:
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

    return center_point, R, [cxTupel, cyTupel, czTupel]
