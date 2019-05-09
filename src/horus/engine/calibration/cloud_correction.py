# -*- coding: utf-8 -*-
# This file is part of the Gryphon Scan Project

__author__ = 'Mikhail N Klimushin aka Night Gryphon <ngryph@gmail.com>'
__copyright__ = 'Copyright (C) 2018 Night Gryphon'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import numpy as np
import cv2
import time

import struct
import math
from scipy.sparse import linalg as splinalg
from scipy import sparse, linalg
import numpy.linalg

from horus import Singleton
from horus.engine.calibration.calibration import CalibrationCancel
from horus.engine.calibration.moving_calibration import MovingCalibration

from horus.gui.util.augmented_view import apply_mask, augmented_pattern_mask, rotatePoint2Plane

from horus.util import profile

import logging
logger = logging.getLogger(__name__)


class CloudCorrectionError(Exception):

    def __init__(self):
        Exception.__init__(self, "CloudCorrectionError")


@Singleton
class CloudCorrection(MovingCalibration):

    def __init__(self):
        MovingCalibration.__init__(self)
        self.image = None
        self.corrections = [None, None]
        self.start_angle = 0 # start calibration from this angle (initial movement). 0 - perpendicular to camera

    def _initialize(self):
        self.image = None
        self.image_capture.stream = False

    def _capture_corners(self, steps = 3):
        corners = []
        for i in xrange(steps):
            image = self.image_capture.capture_pattern()
            if image is not None:
                c = self.image_detection.detect_corners(image)
                if c is not None:
                    corners += [c] # [c.reshape(-1,2)]
        corners = np.float32(corners)
        return image, np.mean(corners, axis=0)
        

    def _move_and_capture(self):
        angle = 0.0

        self.image, corners = self._capture_corners()
        pose    = self.image_detection.detect_pose_from_corners(corners)
        d0, n0, _ = self.image_detection.detect_pattern_plane(pose)

        # choose points for calibration. closest to the focal center are less distorted
        corner_id = []
        for y in xrange(self.pattern.rows):
            for x in xrange(self.pattern.columns):
                i = y*self.pattern.columns + x
                if corners[i][0][0] > self.calibration_data.camera_matrix[0][2]:
                    if i>0:
                        corner_id += [i-1]
                    corner_id += [i]
                    print ("%f - %f - %f" % (corners[i-1][0][0], self.calibration_data.camera_matrix[0][2], corners[i][0][0]) )
                    break
        print(corner_id)

        # find center of mass in world coords
        p0 = corners[tuple(corner_id),0]
        print(p0.shape)
        p_center = self.point_cloud_generation.compute_camera_point_cloud(p0.T, d0, n0)  
        p_center = np.mean(p_center.T, axis = 0)
        print(p_center)

        p0_3d = self.point_cloud_generation.compute_platform_point_cloud(p0.T, None, d0, n0)  

        # measure corrections
        angles = []
        #perfect_points = []
        #clouds = []
        #for index, laser in reversed(list(enumerate(self.calibration_data.laser_planes))):
        for index, laser in enumerate(self.calibration_data.laser_planes):
            # get platform rotation to move points to laser plane position
            l = rotatePoint2Plane(p_center, laser.normal, laser.distance)
            print("-------- Angle: "+str(l))
            angles += [l]
            # calculate perfect point cloud

            # use model coords to rotate cloud to desired position
            #perfect_points = self.point_cloud_generation.compute_point_cloud(np.deg2rad(-l), p0.T, None, d0, n0)

            # Rotate to angle
            c, s = np.cos(-np.deg2rad(-l)), np.sin(-np.deg2rad(-l))
            Rz = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            perfect_points = Rz * p0_3d

            #print( np.round(perfect_points, 3) ) 
        
            # measure real positions
            self.driver.board.motor_move(-angle+l)
            angle = l
            self.driver.board.laser_on(index)
            time.sleep(0.5)
            self.driver.board.lasers_off()

            self.image, corners = self._capture_corners()
            p = corners[tuple(corner_id),0]
            pose    = self.image_detection.detect_pose_from_corners(corners)
            d, n, _ = self.image_detection.detect_pattern_plane(pose)
            
            cloud = self.point_cloud_generation.compute_platform_point_cloud(p.T, None, d, n)
            #print( np.round(cloud, 3) )

            print( np.round(np.array(cloud - perfect_points), 3) )


        return angle

        pose = self.image_detection.detect_pose(image, False)
        plane = self.image_detection.detect_pattern_plane(pose)
        if plane is not None:
            distance, normal, corners = plane
            # Angle between the pattern and the camera
            # measure angle within camera XZ plane. 
            # Negative angle for pattern face turned to the left from camera
            alpha = -np.rad2deg(math.acos(normal[2]/np.linalg.norm( (normal[0], normal[2]) ))) * math.copysign(1, normal[0])
            lasers = np.where(np.logical_and(
                self.laser_calibration_angles[:,0]<alpha,
                self.laser_calibration_angles[:,1]>alpha ))[0]
            if lasers.size > 0:
#                self.image_capture.flush_laser()
#                self.image_capture.flush_laser()
                images = self.image_capture.capture_lasers()[:-1]

                if self.points_image is None:
                    self.points_image = np.zeros(images[0].shape, dtype = "uint8")
                self.image = np.copy(self.points_image)
                #self.image = image
                colors = [(255,255,0), (0,255,255), (255,0,255)]

                for i in lasers:
                    image = images[i]
                    if image is not None:
                        image = self.image_detection.pattern_mask(image, corners)
                      
                        np.maximum(self.image, image, out = self.image)
                      
                        points_2d, image = self.laser_segmentation.compute_2d_points(image)
                        self.points_image[points_2d[1],np.rint(points_2d[0]).astype(int)] = colors[i]
                        #points_2d = self.point_cloud_generation.undistort_points(points_2d)
                        point_3d = self.point_cloud_generation.compute_camera_point_cloud(
                            points_2d, distance, normal)
                        if self._point_cloud[i] is None:
                            self._point_cloud[i] = point_3d.T
                        else:
                            self._point_cloud[i] = np.concatenate(
                                (self._point_cloud[i], point_3d.T))

            else:
                self.image = image
                print("Skip calibration at "+str(alpha))
        else:
            self.image = image

    def _calibrate(self):
        self.has_image = False
        self.image_capture.stream = True

        self.std = [None, None]

        return (False, (None,None,None,None,None,None,None,None,None,None))

        # Compute planes
        for i in xrange(2):
            if self._is_calibrating:
                plane = compute_plane(i, self._point_cloud[i])
                self.distance[i], self.normal[i], self.std[i] = plane

        if self._is_calibrating:
            if self.std[0] < 1.0 and self.std[1] < 1.0 and \
               self.normal[0] is not None and self.normal[1] is not None:
                response = (True, ((self.distance[0], self.normal[0], self.std[0]),
                                   (self.distance[1], self.normal[1], self.std[1]),
                                   self._point_cloud))
            else:
                response = (False, CloudCorrectionError())
        else:
            response = (False, CalibrationCancel())

        self._is_calibrating = False
        self.image = None

        return response

    def accept(self):
        for i in xrange(2):
            self.calibration_data.laser_planes[i].correction = self.corrections[i]


