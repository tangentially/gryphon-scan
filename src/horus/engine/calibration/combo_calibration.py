# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import math
import numpy as np

from horus import Singleton
from horus.engine.calibration.calibration import CalibrationCancel
from horus.engine.calibration.moving_calibration import MovingCalibration
from horus.engine.calibration import laser_triangulation, platform_extrinsics

from horus.util import profile

import logging
logger = logging.getLogger(__name__)


class ComboCalibrationError(Exception):

    def __init__(self):
        Exception.__init__(self, "ComboCalibrationError")


@Singleton
class ComboCalibration(MovingCalibration):

    """Combine Laser Triangulation and Platform Extrinsics calibration in one"""

    def __init__(self):
        self.image = None
        self.has_image = False
        self.laser_calibration_angles = np.float32( [ (-90,90), (-90,90) ])
        MovingCalibration.__init__(self)

    def _initialize(self):
        self.image = None
        self.has_image = True
        self.image_capture.stream = False
        self._point_cloud = [None, None]
        self.x = []
        self.y = []
        self.z = []

    def read_profile(self):
        MovingCalibration.read_profile(self)
        self.laser_calibration_angles = profile.settings['laser_calibration_angles']
        
    def _capture(self, angle):
        image = self.image_capture.capture_pattern()
        pose = self.image_detection.detect_pose(image)
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
                for i in lasers:
                    image = self.image_capture.capture_laser(i)[0]
                    image = self.image_detection.pattern_mask(image, corners)
                    self.image = image
                    points_2d, image = self.laser_segmentation.compute_2d_points(image)
                    point_3d = self.point_cloud_generation.compute_camera_point_cloud(
                        points_2d, distance, normal)
                    if self._point_cloud[i] is None:
                        self._point_cloud[i] = point_3d.T
                    else:
                        self._point_cloud[i] = np.concatenate(
                            (self._point_cloud[i], point_3d.T))
            else:
                self.image = image
                print("Skip laser calibration at "+str(alpha))

            # Platform extrinsics
            pp = (self.pattern.square_width * (self.pattern.rows-1)) * pose[0].T[1] + pose[1].T[0]
            self.x += [pp[0]]
            self.y += [pp[1]]
            self.z += [pp[2]]
        else:
            self.image = image

    def _calibrate(self):
        self.has_image = False
        self.image_capture.stream = True

        # Laser triangulation
        # Save point clouds
        for i in range(2):
            laser_triangulation.save_point_cloud('PC' + str(i) + '.ply', self._point_cloud[i])

        self.distance = [None, None]
        self.normal = [None, None]
        self.std = [None, None]

        # Compute planes
        for i in range(2):
            if self._is_calibrating:
                plane = laser_triangulation.compute_plane(i, self._point_cloud[i])
                self.distance[i], self.normal[i], self.std[i] = plane

        # Platform extrinsics
        self.t = None
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)
        points = list(zip(self.x, self.y, self.z))

        if len(points) > 4:
            # Fitting a plane
            point, normal = platform_extrinsics.fit_plane(points)
            if normal[1] > 0:
                normal = -normal
            # Fitting a circle inside the plane
            center, self.R, circle = platform_extrinsics.fit_circle(point, normal, points)
            # Get real origin
            self.t = center - self.pattern.origin_distance * np.array(normal)

            logger.info("Platform calibration ")
            logger.info(" Translation: " + str(self.t))
            logger.info(" Rotation: " + str(self.R).replace('\n', ''))
            logger.info(" Normal: " + str(normal))

        # Return response
        result = True
        if self._is_calibrating:
            if self.t is not None and \
               np.linalg.norm(self.t - platform_extrinsics.estimated_t) < 100:
                response_platform_extrinsics = (
                    self.R, self.t, center, point, normal, [self.x, self.y, self.z], circle)
            else:
                result = False

            if self.std[0] < 1.0 and self.std[1] < 1.0 and \
               self.normal[0] is not None and self.normal[1] is not None:
                response_laser_triangulation = ((self.distance[0], self.normal[0], self.std[0]),
                                                (self.distance[1], self.normal[1], self.std[1]))
            else:
                result = False

            if result:
                response = (True, (response_platform_extrinsics, response_laser_triangulation))
            else:
                response = (False, ComboCalibrationError())
        else:
            response = (False, CalibrationCancel())

        self._is_calibrating = False
        self.image = None

        return response

    def accept(self):
        for i in range(2):
            self.calibration_data.laser_planes[i].distance = self.distance[i]
            self.calibration_data.laser_planes[i].normal = self.normal[i]
        self.calibration_data.platform_rotation = self.R
        self.calibration_data.platform_translation = self.t
