# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import numpy as np
try:
    import cv2.aruco as aruco
    aruco_present = True
except ImportError:
    aruco_present = False


from horus import Singleton

from horus.util import profile

@Singleton
class Pattern(object):

    def __init__(self):
        self._rows = 0
        self._columns = 0
        self._square_width = 0
        self.origin_distance = 0
        self.border_l = 0
        self.border_r = 0
        self.border_t = 0
        self.border_b = 0

        if aruco_present:
            self.aruco_size = 36
            self.aruco_dict = aruco.DICT_4X4_50 # aruco.DICT_6X6_250

    def read_profile(self):
        self.rows = profile.settings['pattern_rows']
        self.columns = profile.settings['pattern_columns']
        self.square_width = profile.settings['pattern_square_width']
        self.origin_distance = profile.settings['pattern_origin_distance']
        self.border_l = profile.settings['pattern_border_l']
        self.border_r = profile.settings['pattern_border_r']
        self.border_t = profile.settings['pattern_border_t']
        self.border_b = profile.settings['pattern_border_b']

    @property
    def rows(self):
        return self._rows

    @rows.setter
    def rows(self, value):
        value = self.to_int(value)
        if self._rows != value:
            self._rows = value
            self._generate_object_points()

    def set_rows(self, value):
        self.rows = value

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, value):
        value = self.to_int(value)
        if self._columns != value:
            self._columns = value
            self._generate_object_points()

    def set_columns(self, value):
        self.columns = value

    @property
    def square_width(self):
        return self._square_width

    @square_width.setter
    def square_width(self, value):
        value = self.to_float(value)
        if self._square_width != value:
            self._square_width = value
            self._generate_object_points()

    def set_square_width(self, value):
        self.square_width = value

    def _generate_object_points(self):
        objp = np.zeros((self.rows * self.columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.columns, 0:self.rows].T.reshape(-1, 2)
        objp = np.multiply(objp, self.square_width)
        self.object_points = objp

    def set_origin_distance(self, value):
        self.origin_distance = self.to_float(value)

    @staticmethod
    def to_int(value):
        try:
            value = int(value)
            if value > 0:
                return value
            else:
                return 0
        except:
            return 0

    @staticmethod
    def to_float(value):
        try:
            value = float(value)
            if value > 0.0:
                return value
            else:
                return 0.0
        except:
            return 0.0

pattern = Pattern()
