# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'


import numpy as np
import cv2

from horus import Singleton
from horus.engine.calibration.calibration_data import CalibrationData
from horus.engine.driver.driver import Driver


@Singleton
class PointCloudGeneration(object):

    def __init__(self):
        self.calibration_data = CalibrationData()
        self.driver = Driver() # load driver singleton

    def compute_point_cloud(self, theta, points_2d, index):
        # Load calibration values
        R = np.matrix(self.calibration_data.platform_rotation)
        t = np.matrix(self.calibration_data.platform_translation).T
        # Compute platform transformation
        Xwo = self.compute_platform_point_cloud(points_2d, R, t, index)
        # Rotate to world coordinates
        c, s = np.cos(-theta), np.sin(-theta)
        Rz = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        Xw = Rz * Xwo
        # Return point cloud
        if Xw.size > 0:
            return np.array(Xw)
        else:
            return None

    def compute_platform_point_cloud(self, points_2d, R, t, index):
        # Load calibration values
        n = self.calibration_data.laser_planes[index].normal
        d = self.calibration_data.laser_planes[index].distance
        # Camera system
        Xc = self.compute_camera_point_cloud(points_2d, d, n)
        # Compute platform transformation
        return R.T * Xc - R.T * t


    def undistort_points(self, points_2d, camera_shape = None): #, keepSize = True):
        # camera_shape = image.shape[::-1] 
        # correct camera distortion
        print("---------- undistort_points ------------")
        print(points_2d.T)

        #pts = np.asarray(tuple(points_2d)).transpose()
        #pts = np.expand_dims(pts, axis=0)

        #print(pts.shape)
        #print(pts[0][0])
        #rs = cv2.undistortPoints(pp, self.calibration_data.camera_matrix, self.calibration_data.distortion_vector)
        #print(rs)

        if not camera_shape:
            camera_shape = self.driver.camera.get_resolution()
	cam = self.calibration_data.camera_matrix
        d = self.calibration_data.distortion_vector

        #pts = np.asarray(points, dtype=np.float32)
        #if pts.ndim == 2:
        #    pts = np.expand_dims(pts, axis=0)

        #newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cam, d, camera_shape, 1, camera_shape, 0)
        #print(roi)

	#xx, yy = roi[:2]
        #u,v = points_2d
        #pts = np.asarray((u - xx, v - yy)).transpose()
        pts = np.asarray(tuple(points_2d)).transpose()
        pts = np.expand_dims(pts, axis=0)

        #print(pts)
        #if not keepSize:
        #    xx, yy = roi[:2]
        #    pts[0, 0] -= xx
        #    pts[0, 1] -= yy

        #rs = cv2.undistortPoints(pts, cam, d, P=newCameraMatrix)[0]
        rs = cv2.undistortPoints(pts, cam, d, P=cam)[0]
        print(rs)
        return rs


    def compute_camera_point_cloud(self, points_2d, d, n):
        # Load calibration values
        fx = self.calibration_data.camera_matrix[0][0]
        fy = self.calibration_data.camera_matrix[1][1]
        cx = self.calibration_data.camera_matrix[0][2]
        cy = self.calibration_data.camera_matrix[1][2]
        # Compute projection point
        u, v = points_2d
        x = np.concatenate( ((u - cx) / fx, (v - cy) / fy, np.ones(len(u))) ).reshape(3, len(u))
        # Compute laser intersection
        return d / np.dot(n, x) * x
