# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np
from distutils.version import LooseVersion

from horus import Singleton
from horus.engine.calibration.calibration import Calibration


class CameraIntrinsicsError(Exception):

    def __init__(self):
        Exception.__init__(self, "CameraIntrinsicsError")


@Singleton
class CameraIntrinsics(Calibration):

    """Camera calibration algorithms, based on [Zhang2000] and [BouguetMCT]:

            - Camera matrix
            - Distortion vector
    """

    def __init__(self):
        Calibration.__init__(self)
        self.shape = None
        self.camera_matrix = None
        self.distortion_vector = None
        self.image_points = []
        self.object_points = []

    def _start(self):
        ret, error, cmat, dvec, rvecs, tvecs = self.calibrate()

        if ret:
            self.camera_matrix = cmat
            self.distortion_vector = dvec
            response = (True, (error, cmat, dvec, rvecs, tvecs))
        else:
            response = (False, CameraIntrinsicsError)

        self._is_calibrating = False

        if self._after_callback is not None:
            self._after_callback(response)

    def capture(self, image = None):
        if self.driver.is_connected:
            if image is None:
                image = self.image_capture.capture_pattern()

            self.shape = image[:, :, 0].shape
            corners = self.image_detection.detect_corners(image)
            if corners is not None:
                if len(self.object_points) < 15:
                    self.image_points.append(corners)
                    self.object_points.append(self.pattern.object_points)
                    return image

    def calibrate(self):
        # https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/camera%20distortion.pdf
        error = 0
        #ret, cmat, dvec, rvecs, tvecs = cv2.calibrateCamera(
        #    self.object_points, self.image_points, self.shape, None, None)

        # use current values to start estimation
        cmat0 = self.calibration_data.camera_matrix
        dvec0 = self.calibration_data.distortion_vector
        print(cmat0)
        if LooseVersion(cv2.getVersionString()) > LooseVersion("3.0.0"):
            flags = cv2.CALIB_USE_INTRINSIC_GUESS
        else:
            flags = cv2.CV_CALIB_USE_INTRINSIC_GUESS

        ret, cmat, dvec, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, self.shape, cmat0, dvec0, 
            flags=flags)

        # TODO More Accurate Camera Calibration with Imperfect Planar Target
        # https://github.com/opencv/opencv/pull/12772
        # https://github.com/xoox/calibrel/blob/master/test/test_calibrel.cpp
        # https://docs.opencv.org/4.1.0/d9/d0c/group__calib3d.html#ggae515fd11d4c0e5b4162440eaf2094e02a5c59485f1b5391cb3d7b2bfb1b7079a7
        # Distortion
        # https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/camera%20distortion.pdf
        # samples\cpp\calibration.cpp
        # rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
        #                    cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
        #                    flags | CALIB_FIX_K3 | CALIB_USE_LU);

        if LooseVersion(cv2.getVersionString()) > LooseVersion("4.0.0"):
            newObjPoints = np.array([], dtype=np.float64)
            '''
            # https://github.com/opencv/opencv/issues/14469
            print("-1-")
            newObjPoints = np.float64(self.object_points)
            ret, cmat1, dvec1, rvecs1, tvecs1, newObjPoints1 = cv2.calibrateCameraRO(
                        self.object_points, self.image_points, self.shape, -1,
                        cmat, dvec, rvecs, tvecs, None, cv2.CALIB_USE_INTRINSIC_GUESS, 
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,  2.2204460492503131e-16)  );
            print("-2-")
            #self.object_points[s.boardSize.width - 1].x = self.object_points[0].x + grid_width;
            ret, cmat2, dvec2, rvecs2, tvecs2, newObjPoints2 = cv2.calibrateCameraRO(
                        self.pattern.object_points, self.image_points, self.shape[::-1], int(self.pattern.columns-1),
                        cmat1, dvec1, rvecs1, tvecs1, newObjPoints1, cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_USE_LU, 
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,  2.2204460492503131e-16)  );
#            ret, cmat2, dvec2, rvecs2, tvecs2, newObjPoints2 = cv2.calibrateCameraRO(
#                        self.object_points, self.image_points, self.shape, self.pattern.columns-1,
#                        cmat, dvec, rvecs=rvecs, tvecs=tvecs, newObjPoints= newObjPoints,
#                        flags= cv2.CALIB_USE_INTRINSIC_GUESS,
#                        criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,  2.2204460492503131e-16) ); # | CALIB_USE_LU
            '''
        if ret:
            # Compute calibration error
            for i in range(len(self.object_points)):
                imgpoints2, _ = cv2.projectPoints(
                    self.object_points[i], rvecs[i], tvecs[i], cmat, dvec)
                error += cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            error /= len(self.object_points)

        return ret, error, np.round(cmat, 3), np.round(dvec.ravel(), 3), rvecs, tvecs

    def reset(self):
        self.image_points = []
        self.object_points = []

    def accept(self):
        self.calibration_data.camera_matrix = self.camera_matrix
        self.calibration_data.distortion_vector = self.distortion_vector

    def cancel(self):
        pass
