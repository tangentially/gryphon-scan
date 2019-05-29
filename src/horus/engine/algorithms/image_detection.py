# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np

from horus import Singleton
from horus.engine.calibration.pattern import pattern
from horus.engine.calibration.calibration_data import calibration_data

from horus.gui.util.augmented_view import augmented_pattern_mask

@Singleton
class ImageDetection(object):

    def __init__(self):
        #self.chessboard_mask = None

        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)


    def detect_pattern(self, image, fast=True):
        corners = self._detect_chessboard(image, fast)
        if corners is not None:
            image = self.draw_pattern(image, corners)
        return image


    def draw_pattern(self, image, corners):
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.drawChessboardCorners(
                image, (pattern.columns, pattern.rows), corners, True)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def detect_corners(self, image, fast=True):
        corners = self._detect_chessboard(image, fast)
        return corners

    def detect_pose(self, image, fast=True):
        corners = self._detect_chessboard(image, fast)
        return self.detect_pose_from_corners(corners)

    def detect_pose_from_corners(self, corners):
        if corners is not None:
            ret, rvecs, tvecs = cv2.solvePnP(
                pattern.object_points, corners,
                calibration_data.camera_matrix,
                calibration_data.distortion_vector)
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

    def _detect_chessboard(self, image, fast=True): 
        if image is not None:
            if pattern.rows > 2 and pattern.columns > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                fl = 0 #cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                if fast:
                    fl = fl | cv2.CALIB_CB_FAST_CHECK
                ret, corners = cv2.findChessboardCorners(
                    gray, (pattern.columns, pattern.rows), flags= fl)
                if ret:
#                    self.chessboard_mask = cv2.threshold(
#                        gray, gray.max() / 2, 255, cv2.THRESH_BINARY)[1]
                    v1 = abs(corners[0][0] - corners[pattern.columns-1][0])/(pattern.columns-1)
                    v2 = abs(corners[0][0] - corners[pattern.columns*(pattern.rows-1)][0])/(pattern.rows-1)
                    w = int(max(v1[0], v2[0])*0.6)
                    h = int(max(v1[1], v2[1])*0.6)
                    cv2.cornerSubPix(gray, corners, (w, h), (1, 1), self._criteria) # (11,11), (-1,-1)
                    return corners
#                else:
#                    self.chessboard_mask = None


