# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'


import md5
import cv2
import numpy as np

from horus import Singleton

from horus.util import profile
from horus.engine.driver.driver import Driver

class LaserPlane(object):

    def __init__(self, name):
        self.normal = None
        self.distance = None
        self.name = name
        self.calibration_segmentation_profile = 0
        self.capture_segmentation_profile = 0

    def is_empty(self):
        if self.distance is None or self.normal is None:
            return True
        if self.distance == 0.0 or np.all(self.normal == 0.0):
            return True
        return False

    def read_profile(self):
        self.distance = profile.settings['distance_'+self.name]
        self.normal   = profile.settings['normal_'+self.name]

    def save_profile(self):
        profile.settings['distance_'+self.name] = self.distance
        profile.settings['normal_'+self.name]   = self.normal

        profile.settings['laser_triangulation_hash'] = self.md5_hash()


@Singleton
class CalibrationData(object):

    def __init__(self):
        self.width = 0
        self.height = 0

        self._camera_matrix = None
        self._distortion_vector = None
        self._roi = None
        self._dist_camera_matrix = None
        self._weight_matrix = None

        self._md5_hash = None

        self.laser_planes = [LaserPlane('left'), LaserPlane('right')]
        self.platform_rotation = None
        self.platform_translation = None

    def read_profile_camera(self):
        driver = Driver() # load driver singleton
        width, height = driver.camera.get_resolution()
        print("calibration_data.read_profile_camera: camera res = "+str(driver.camera.get_resolution()))
        self.set_resolution(width, height)
        self.camera_matrix = profile.settings['camera_matrix']
        self.distortion_vector = profile.settings['distortion_vector']

    def read_profile_laser(self):
        for l in self.laser_planes:
            l.read_profile()

    def save_profile_laser(self):
        for l in self.laser_planes:
            l.save_profile()

        #profile.settings['laser_triangulation_hash'] = self.md5_hash()

    def read_profile_platform(self):
        self.platform_rotation = profile.settings['rotation_matrix']
        self.platform_translation = profile.settings['translation_vector']

    def read_profile_calibration(self):
        self.read_profile_laser()
        self.read_profile_platform()

    def set_resolution(self, width, height):
        if self.width != width or self.height != height:
            self.width = width
            self.height = height
            self._compute_weight_matrix()

    @property
    def camera_matrix(self):
        return self._camera_matrix

    @camera_matrix.setter
    def camera_matrix(self, value):
        self._camera_matrix = value
        self._compute_dist_camera_matrix()

    @property
    def distortion_vector(self):
        return self._distortion_vector

    @distortion_vector.setter
    def distortion_vector(self, value):
        self._distortion_vector = value
        self._compute_dist_camera_matrix()

    @property
    def roi(self):
        return self._roi

    @property
    def dist_camera_matrix(self):
        return self._dist_camera_matrix

    @property
    def weight_matrix(self):
        return self._weight_matrix

    def _compute_dist_camera_matrix(self):
        if self._camera_matrix is not None and self._distortion_vector is not None:
            self._dist_camera_matrix, self._roi = cv2.getOptimalNewCameraMatrix(
                self._camera_matrix, self._distortion_vector,
                (int(self.width), int(self.height)), alpha=1)
            self._md5_hash = md5.new()
            self._md5_hash.update(self._camera_matrix)
            self._md5_hash.update(self._distortion_vector)
            self._md5_hash = self._md5_hash.hexdigest()

    def _compute_weight_matrix(self):
        self._weight_matrix = np.array((np.matrix(np.linspace(0, self.width - 1, self.width)).T *
                                        np.matrix(np.ones(self.height))).T)

    def check_camera_calibration(self):
        if self.camera_matrix is None or self.distortion_vector is None:
            return False
        return True

    def check_lasers_calibration(self):
        for plane in self.laser_planes:
            if plane is None or plane.is_empty():
                return False
        return True

    def check_platform_calibration(self):
        if self.platform_rotation is None or self.platform_translation is None:
            return False
        if self._is_zero(self.platform_rotation) or self._is_zero(self.platform_translation):
            return False
        return True

    def check_calibration(self):
        return self.check_camera_calibration() and \
            self.check_lasers_calibration() and \
            self.check_platform_calibration()

    def _is_zero(self, array):
        return np.all(array == 0.0)

    def md5_hash(self):
        return self._md5_hash
