# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
try:
    import cv2.aruco as aruco
    aruco_present = True
except ImportError:
    aruco_present = False


import numpy as np

from horus import Singleton
from horus.engine.calibration.pattern import Pattern
from horus.engine.calibration.calibration_data import CalibrationData

from horus.gui.util.augmented_view import augmented_pattern_mask

@Singleton
class ImageDetection(object):

    def __init__(self):
        self.pattern = Pattern()
        self.calibration_data = CalibrationData()
        self.chessboard_mask = None

        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if aruco_present:
            self.aruco_dict = aruco.Dictionary_get(self.pattern.aruco_dict)
            # https://docs.opencv.org/3.4.3/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
            self.aruco_parameters = aruco.DetectorParameters_create()
            self.aruco_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG # aruco.CORNER_REFINE_SUBPIX


    def detect_pattern(self, image):
        corners = self._detect_chessboard(image)
        if corners is not None:
            image = self.draw_pattern(image, corners)
        return image

    def draw_pattern(self, image, corners):
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.drawChessboardCorners(
                image, (self.pattern.columns, self.pattern.rows), corners, True)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def detect_corners(self, image):
        corners = self._detect_chessboard(image)
        return corners

    def detect_pose(self, image):
        corners = self._detect_chessboard(image)
        return self.detect_pose_from_corners(corners)

    def detect_pose_from_corners(self, corners):
        if corners is not None:
            ret, rvecs, tvecs = cv2.solvePnP(
                self.pattern.object_points, corners,
                self.calibration_data.camera_matrix, self.calibration_data.distortion_vector)
            if ret:
                return (cv2.Rodrigues(rvecs)[0], tvecs, corners)

    def detect_pattern_plane(self, pose):
        if pose is not None:
            R = pose[0]
            t = pose[1].T[0]
            c = pose[2]
            n = R.T[2]
            d = np.dot(n, t)
            return (d, n, c)

    def pattern_mask(self, image, corners):
        if image is not None:
            h, w, d = image.shape
            if corners is not None:
                mask = augmented_pattern_mask(image, corners)
                image = cv2.bitwise_and(image, image, mask=mask)
#                if self.chessboard_mask is not None:
#                    image = cv2.bitwise_and(image, image, mask=self.chessboard_mask)
        return image

    def _detect_chessboard(self, image, retry = True):
        if image is not None:
            if self.pattern.rows > 2 and self.pattern.columns > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray, (self.pattern.columns, self.pattern.rows), flags=cv2.CALIB_CB_FAST_CHECK)
                if retry and not ret:
                    ret, corners = cv2.findChessboardCorners(
                        gray, (self.pattern.columns, self.pattern.rows), flags=0)
                if ret:
                    self.chessboard_mask = cv2.threshold(
                        gray, gray.max() / 2, 255, cv2.THRESH_BINARY)[1]
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._criteria)
                    return corners
                else:
                    self.chessboard_mask = None


    def aruco_detect(self, image):
        if not aruco_present:
            return False

        if image is None:
            return False

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_parameters)

        return (corners, ids)


    def aruco_pose_from_corners(self, corners):
        if not aruco_present:
            return False

        #Estimate pose of marker and return the values rvet and tvec---different from camera coefficients
        rvecs, tvecs,_ = aruco.estimatePoseSingleMarkers(corners, self.pattern.aruco_size, 
                          self.calibration_data.camera_matrix, self.calibration_data.distortion_vector) 

        return (rvecs, tvecs)


    def aruco_draw_markers(self, image, corners, ids):
        rvecs = False
        tvecs = False
        if aruco_present and \
           image is not None and \
           ids.size > 0:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image = aruco.drawDetectedMarkers(image, corners, ids)
            
            rvecs, tvecs = self.aruco_pose_from_corners(corners)
            for idx, aid in enumerate(ids):
                image = aruco.drawAxis(image, 
                         self.calibration_data.camera_matrix, self.calibration_data.distortion_vector, 
                         rvecs[idx], tvecs[idx], 
                         self.pattern.aruco_size / 2)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (image, rvecs, tvecs)




