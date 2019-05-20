# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'


import numpy as np
import cv2

from horus import Singleton
from horus.engine.calibration.calibration_data import CalibrationData


@Singleton
class PointCloudGeneration(object):

    def __init__(self):
        self.calibration_data = CalibrationData()

    def compute_point_cloud(self, theta, points_2d, index, d = None, n = None, M = None):
        # compute point cloud in model coords
        #   theta - rad, platform position
        #   points_2d = [u,v]
        #   d,n - projection plane
        #   M - cloud correction matrix

        # Compute in turntable coords
        Xwo = self.compute_platform_point_cloud(points_2d, index, d, n)

        # Rotate to model coordinates
        c, s = np.cos(-theta), np.sin(-theta)
        Rz = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        Xw = Rz * Xwo
        '''
        # Correction
        if M is None and index is not None:
            M = self.calibration_data.laser_planes[index].correction
        if M is not None:
            M = np.matrix(M)
            Xw = np.insert( Xw, 3, [1.], axis=0)
            Xw = (M * Xw)
        '''
        # Return point cloud
        if Xw.size > 0:
            return np.array(Xw)
        else:
            return None

    def compute_platform_point_cloud(self, points_2d, index, d = None, n = None):
        # compute point cloud in platform coords
        #   points_2d = [u,v]
        #   R, t has to be np.matrix

        # Load laser plane position
        if n is None and index is not None:
            n = self.calibration_data.laser_planes[index].normal
        assert n is not None, "Plane normal not defined"

        if d is None and index is not None:
            d = self.calibration_data.laser_planes[index].distance
        assert n is not None, "Plane distance not defined"

        # Load calibration values
        R = np.matrix(self.calibration_data.platform_rotation)
        t = np.matrix(self.calibration_data.platform_translation).T

        # Camera system
        Xc = self.compute_camera_point_cloud(points_2d, d, n)
        # Compute platform transformation
        return R.T * Xc - R.T * t


    def undistort_points(self, points_2d):
        # correct camera distortion
        #   points_2d = [u,v]

        if len(points_2d[0]) == 0:
            return points_2d

	cam = self.calibration_data.camera_matrix
        d = self.calibration_data.distortion_vector

        pts = np.expand_dims(np.float32(points_2d).transpose(), axis=1)

        rs = cv2.undistortPoints(pts, cam, d, P=cam)
        rs = tuple(rs.reshape(-1,2).T)
        return rs


    def compute_camera_point_cloud_horus(self, points_2d, d, n):
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


    def compute_camera_point_cloud(self, points_2d, d, n):
        # compute point cloud in world coords
        #   points_2d = [u,v]

        points_2d = np.float32(points_2d)
        assert points_2d.shape[0] == 2, "points_2d should be [u,v] array!!!"

        if len(points_2d[0]) == 0:
            return np.empty( (3,0), dtype=np.float32 )

        n = np.float32(n)
        assert n.shape == (3,), "n should be (3,) vector!!!" 

        # Load calibration values
	cam = self.calibration_data.camera_matrix
        dist = self.calibration_data.distortion_vector

        # Compute projection point
        pts = np.expand_dims(np.float32(points_2d).transpose(), axis=1)
        x = cv2.undistortPoints(pts, cam, dist).reshape(-1,2).T # normalized results [u,v]  ( [[(x-cx)/fx], [(y-cy)/fy]] )

        # Compute laser intersection
        x = np.insert( x, 2, [1.], axis=0) # [u,v,1]
        return d / np.dot(n, x).reshape(1,-1) * x # [X,Y,Z]

