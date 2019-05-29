# This file is part of the Gryphon Scan Project
__author__ = 'Mikhail N Klimushin aka Night Gryphon <ngryph@gmail.com>'
__copyright__ = 'Copyright (C) 2019 Night Gryphon'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'


import cv2
try:
    import cv2.aruco as aruco
    aruco_present = True
except ImportError:
    aruco_present = False


import numpy as np

from horus import Singleton
from horus.engine.calibration.calibration_data import calibration_data
from horus.engine.calibration.pattern import pattern

@Singleton
class ArucoDetection(object):

    def __init__(self):
        if not aruco_present:
            return None

        self.aruco_dict = aruco.Dictionary_get(pattern.aruco_dict)
        # https://docs.opencv.org/3.4.3/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
        self.aruco_parameters = aruco.DetectorParameters_create()
        self.aruco_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG # aruco.CORNER_REFINE_SUBPIX


    def aruco_detect(self, image):
        if not aruco_present:
            return (None, None)

        if image is None:
            return (None, None)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_parameters)

        return (corners, ids)


    def aruco_pose_from_corners(self, corners):
        if not aruco_present:
            return (None, None)

        #Estimate pose of marker and return the values rvet and tvec---different from camera coefficients
        rvecs, tvecs,_ = aruco.estimatePoseSingleMarkers(corners, pattern.aruco_size, 
                          calibration_data.camera_matrix, calibration_data.distortion_vector) 

        return (rvecs, tvecs)


    def aruco_draw_markers(self, image, corners, ids):
        rvecs = None
        tvecs = None
        if aruco_present and \
           image is not None and \
           ids.size > 0:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image = aruco.drawDetectedMarkers(image, corners, ids)
            
            rvecs, tvecs = self.aruco_pose_from_corners(corners)
            for idx, aid in enumerate(ids):
                image = aruco.drawAxis(image, 
                         calibration_data.camera_matrix, 
                         calibration_data.distortion_vector, 
                         rvecs[idx], tvecs[idx], 
                         pattern.aruco_size / 2)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (image, rvecs, tvecs)

aruco_detection = ArucoDetection()
