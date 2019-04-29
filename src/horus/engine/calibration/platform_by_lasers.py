# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np

from horus import Singleton
from horus.engine.calibration.calibration import Calibration


class PlatformByLasersError(Exception):

    def __init__(self):
        Exception.__init__(self, "PlatformByLasersError")


@Singleton
class PlatformByLasers(Calibration):

    """
       Calibrate platform normal by laser lines
    """           

    def __init__(self):
        Calibration.__init__(self)
        self._point_cloud = None

        self.shape = None
        self.camera_matrix = None
        self.distortion_vector = None
        self.image_points = []
        self.object_points = []

    def _start(self):
        if self.driver.is_connected and \
           self.calibration_data.check_calibration():
            # machine has to be initially calibrated 
            # laser and camera has to be well calibrated
            # will try to use lasers to enhance turntable calibration

            # build work area mask for turntable surface
            mask = np.zeros(images[0].shape[0:2], dtype = "uint8")
            if profile.settings['use_roi']:
                bounds = profile.get_roi_size_polygons()
            else:
                bounds = profile.get_machine_size_polygons()
            platform_border = []
            for p in bounds[0]:
                platform_border.append( ( p[0], p[1], 0) )
            platform_border = np.float32(platform_border)

            # project platform border to mask
            p, jac = cv2.projectPoints(platform_border, \
                self.calibration_data.platform_rotation, \
                self.calibration_data.platform_translation, \
                self.calibration_data.camera_matrix, \
                self.calibration_data.distortion_vector)
            p = np.int32([p])
            cv2.fillPoly(mask, p, 1)

            # capture laser lines
            images = self.image_capture.capture_lasers()[:-1] # the last image is background. skip it
            point_cloud = None
            for idx,image in enumerate(images):
                image = apply_mask(image, mask)
                points_2d, image = self.laser_segmentation.compute_2d_points(image)
                points_2d = self.point_cloud_generation.undistort_points(points_2d)
                point_3d = self.point_cloud_generation.compute_camera_point_cloud(
                    points_2d, 
                    self.calibration_data.laser_planes[idx].distance, 
                    self.calibration_data.laser_planes[idx].normal)
                if point_cloud is None:
                    point_cloud = point_3d.T
                else:
                    point_cloud = np.concatenate(point_cloud, point_3d.T)

                dist, norm, std = compute_plane(i, point_cloud)

                if std < 1.0 and norm is not None:
                response = (True, (dist, norm, std))
            else:
                response = (False, PlatformByLasersError())

            self._is_calibrating = False

            if self._after_callback is not None:
                self._after_callback(response)


    def reset(self):
        self.image_points = []
        self.object_points = []

    def accept(self):
        self.calibration_data.camera_matrix = self.camera_matrix
        self.calibration_data.distortion_vector = self.distortion_vector

    def cancel(self):
        pass
