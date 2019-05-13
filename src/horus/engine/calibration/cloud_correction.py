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

from horus.gui.util.augmented_view import augmented_pattern_mask
from horus.util.gryphon_util import rotatePoint2Plane, \
    rigid_transform_3D, PointOntoLine, capture_precise_corners

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

        self.p0_3d = None
        self.angles = None
        self.clouds = None

    def _move_and_capture(self):
        angle = 0.0
        ncaptures = 1

        total_captures = 1 + ncaptures*len(self.calibration_data.laser_planes)
        progress = 0
        if self._progress_callback is not None:
            self._progress_callback(100*progress/total_captures)

        print("--- Measure center")
        self.image, corners, std = capture_precise_corners(13)
        progress += 1
        if self._progress_callback is not None:
            self._progress_callback(100*progress/total_captures)

        print("Capture STD: {0:f}".format(std))
        pose    = self.image_detection.detect_pose_from_corners(corners)
        d0, n0, _ = self.image_detection.detect_pattern_plane(pose)

        # choose points for calibration. closest to the focal center are less distorted
        print("--- Points selection")
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

        p0 = corners[tuple(corner_id),0]

        # reference point cloud
        #self.p0_3d = np.insert(self.point_cloud_generation.compute_platform_point_cloud(p0.T, None, d0, n0), 0, [0], axis=1)
        self.p0_3d = self.point_cloud_generation.compute_platform_point_cloud(p0.T, None, d0, n0)
        print(self.p0_3d)

        # calulate angles:
        #   - find center of mass in world coords
        p_center = self.point_cloud_generation.compute_camera_point_cloud(p0.T, d0, n0)  
        p_center = np.mean(p_center.T, axis = 0)

        #   - calculate platform rotations to move center of mass to laser planes
        self.angles = []
        self.clouds = []
        for index, laser in enumerate(self.calibration_data.laser_planes):
            # get platform rotation to move points to laser plane position
            l = rotatePoint2Plane(p_center, laser.normal, laser.distance)
            self.angles += [l]
            self.clouds += [[]]


        # measure actual positions
        #self.clouds = np.empty((len(self.calibration_data.laser_planes),0))
        for i in xrange(ncaptures):
            if not self._is_calibrating:
                break

            #for index, laser in reversed(list(enumerate(self.calibration_data.laser_planes))):
            for index, l in enumerate(self.angles):
                l = self.angles[index]
                print("--- Measurement {0} Angle: {1:f}".format(i+1,l))
        
                # measure real positions
                self.driver.board.motor_move(-angle+l)
                angle = l
                #self.driver.board.laser_on(index)
                #time.sleep(0.5)
                #self.driver.board.lasers_off()
        
                self.image, corners, std = capture_precise_corners(13)
                progress += 1
                if self._progress_callback is not None:
                    self._progress_callback(100*progress/total_captures)

                print("Capture STD: {0:f}".format(std))
                p = corners[tuple(corner_id),0]
                pose    = self.image_detection.detect_pose_from_corners(corners)
                d, n, _ = self.image_detection.detect_pattern_plane(pose)
                
                #cloud = np.insert(self.point_cloud_generation.compute_platform_point_cloud(p.T, None, d, n), 0, [0], axis=1)
                cloud = self.point_cloud_generation.compute_platform_point_cloud(p.T, None, d, n)
                self.clouds[index] += [cloud]
                #print("Points STD: {0:f}".format(np.max(np.std(self.clouds[index], axis=0)) ) )

        return angle


    def _calibrate(self):
        self.image_capture.stream = True

        if not self._is_calibrating:
            return (False, (None,None,None,None,None))

        # calculate correction matrices
        self.corrections = []
        err = []
        for index, l in enumerate(self.angles):
            #l = self.angles[index]
            print(">>> Laser {0}, Angle: {1:f}".format(index,l))
            print("Points STD: {0:f}".format(np.max(np.std(self.clouds[index], axis=0)) ) )
            cloud = np.mean(self.clouds[index], axis = 0)
        
            # calculate perfect point cloud
            # use model coords to rotate cloud to desired position
            #perfect_points = self.point_cloud_generation.compute_point_cloud(np.deg2rad(-l), p0.T, None, d0, n0)

            # Rotate to angle
            c, s = np.cos(-np.deg2rad(-l)), np.sin(-np.deg2rad(-l))
            Rz = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            perfect_points = Rz * self.p0_3d
            print("Points displacements from perfect_points: ")

            #print( np.round(np.array(cloud - perfect_points), 3) )
            delta = np.linalg.norm(cloud - perfect_points, axis=0)
            print( np.round(delta, 3) )
            print( np.round(np.max(delta), 3) )
            print( np.round(np.mean(delta), 3) )
            '''
            ret, M, inliers = cv2.estimateAffine3D(cloud.T, perfect_points.T, None, None, \
                                    ransacThreshold = 0.1, confidence = 0.99)
            if ret:
                M = np.matrix(M)
                self.corrections += [M]
                #print(M)
                cloud = np.insert( cloud, 3, [1.], axis=0)
                #print(cloud.T[0])
                #print(perfect_points.T[0])
                corrected = M * cloud
                #print( corrected.shape )
                #print( corrected.T[0] )
                print("With correction: ")
                delta = np.linalg.norm(M * cloud - perfect_points, axis=0)
                print( np.round(delta, 3) )
                print( np.round(np.max(delta), 3) )
                print( np.round(np.mean(delta), 3) )
                err += [np.mean(delta)]
            else:
                self.corrections += [None]
                err += [None]
            '''
            M,t,_,_ = rigid_transform_3D(cloud.T, perfect_points.T)
            print(M)
            print(t)
            print(cloud.shape)
            corrected = [np.matmul(M,v) + t for v in cloud.T]
            print("With correction: ")
            print(corrected)
            delta = np.linalg.norm(M * cloud - perfect_points, axis=0)
            print( np.round(delta, 3) )
            print( np.round(np.max(delta), 3) )
            print( np.round(np.mean(delta), 3) )
            err += [np.mean(delta)]


        self._is_calibrating = False
        self.image = None

        return (True, (self.corrections, err, self.p0_3d, self.angles, self.clouds))


    def accept(self):
        for i, laser in enumerate(self.calibration_data.laser_planes):
            laser.correction = self.corrections[i]


