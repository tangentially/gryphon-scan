# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Mikhail N Klimushin aka Night Gryphon <ngryph@gmail.com>'
__copyright__ = 'Copyright (C) 2018 Night Gryphon'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np
from horus.util import profile
from horus.gui.engine import platform_extrinsics


class AugmentedView():

    def __init__(self):
        self.platform_points = None
        self.platform_border = None

    # Platform visualization cache shapes
    def init_platform_draw(self):
        r = profile.settings['platform_markers_diameter']/2 #138
        h = profile.settings['platform_markers_z']
        rr = r * np.sin(np.deg2rad(45))
        self.platform_points = np.float32( [(0,0,0),(r,0,h),(-r,0,h),(0,r,h),(0,-r,h),(rr,rr,h),(rr,-rr,h),(-rr,rr,h),(-rr,-rr,h)])

        h = profile.settings['machine_shape_z']
	polys = profile.get_machine_size_polygons(profile.settings["machine_shape"])
        self.platform_border = []
        for p in polys[0]:
            self.platform_border.append( ( p[0], p[1], h) )
        self.platform_border = np.float32(self.platform_border)

    # Platform visualization draw
    def draw_platform(self, image, mask=False):
        if self.platform_points is None:
            self.init_platform_draw()

        if platform_extrinsics.calibration_data.platform_rotation is not None and \
           platform_extrinsics.calibration_data.platform_translation is not None:
            #cv2.circle(image, (50,50), 5, (0,0,255), -1)
            # platform border
            p, jac = cv2.projectPoints(self.platform_border, \
                platform_extrinsics.calibration_data.platform_rotation, \
                platform_extrinsics.calibration_data.platform_translation, \
                platform_extrinsics.calibration_data.camera_matrix, \
                platform_extrinsics.calibration_data.distortion_vector)
            p = np.int32([p])
            cv2.polylines(image, p, True, (0,255,0), 2)

            # marker positions
            p, jac = cv2.projectPoints(self.platform_points, \
                platform_extrinsics.calibration_data.platform_rotation, \
                platform_extrinsics.calibration_data.platform_translation, \
                platform_extrinsics.calibration_data.camera_matrix, \
                platform_extrinsics.calibration_data.distortion_vector)
            p = np.int32(p).reshape(-1,2)
            for pp in p:
                cv2.circle(image, tuple(pp), 5, (0,0,255), -1)
