# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import time
import Queue
import numpy as np
import datetime
import struct
import os

from horus import Singleton
from horus.engine.scan.scan import Scan
from horus.engine.scan.scan_capture import ScanCapture
from horus.engine.scan.current_video import CurrentVideo
from horus.engine.calibration.calibration_data import CalibrationData

from horus.util import profile

import logging
logger = logging.getLogger(__name__)

import platform
system = platform.system()


class ScanError(Exception):

    def __init__(self):
        Exception.__init__(self, "Scan Error")


@Singleton
class CiclopScan(Scan):
    """Perform Ciclop scanning algorithm:

        - Capture Thread: capture raw images and manage motor and lasers
        - Process Thread: compute 3D point cloud from raw images
    """

    def __init__(self):
        Scan.__init__(self)
        self.image = None
        self.current_video = CurrentVideo()
        self.calibration_data = CalibrationData()
        self.capture_texture = True
        self.laser = [True, True]
        self.move_motor = True
        self.motor_step = 0
        self.motor_speed = 0
        self.motor_acceleration = 0
        self.color = (0, 0, 0)

        self._theta = 0
        self._count = 0
        self._debug = False
        self._bicolor = False
        self._scan_sleep = 0.05
        self._captures_queue = Queue.Queue(10)
        self.point_cloud_callback = None

        self.ph_save_enable = False
        self.ph_save_folder = 'photo/'
        self.ph_save_divider = 1

    def read_profile(self):
        self.capture_texture = profile.settings['capture_texture']
        use_laser = profile.settings['use_laser']
        self.set_use_left_laser(use_laser == 'Left' or use_laser == 'Both')
        self.set_use_right_laser(use_laser == 'Right' or use_laser == 'Both')
        self.motor_step = profile.settings['motor_step_scanning']
        self.motor_speed = profile.settings['motor_speed_scanning']
        self.motor_acceleration = profile.settings['motor_acceleration_scanning']
        self.color = struct.unpack(
            'BBB', profile.settings['point_cloud_color'].decode('hex'))
        self.set_scan_sleep(profile.settings['scan_sleep'])

        self.ph_save_enable = profile.settings['ph_save_enable']
        self.ph_save_folder = profile.settings['ph_save_folder']
        self.ph_save_divider = profile.settings['ph_save_divider']

    def set_capture_texture(self, value):
        self.capture_texture = value

    def set_use_left_laser(self, value):
        self.laser[0] = value

    def set_use_right_laser(self, value):
        self.laser[1] = value

    def set_move_motor(self, value):
        self.move_motor = value

    def set_motor_step(self, value):
        self.motor_step = value

    def set_motor_speed(self, value):
        self.motor_speed = value

    def set_motor_acceleration(self, value):
        self.motor_acceleration = value

    def set_debug(self, value):
        self._debug = value

    def set_scan_sleep(self, value):
        self._scan_sleep = value / 1000.

    def _initialize(self):
        self.image = None
        self.image_capture.stream = False
        self._theta = 0
        self._count = 0
        self._progress = 0
        self._captures_queue.queue.clear()
        self._begin = time.time()

        # Setup console
        logger.info("Start scan")
        if self._debug and system == 'Linux':
            string_time = str(datetime.datetime.now())[:-3] + " - "
            print string_time + " elapsed progress: 0 %"
            print string_time + " elapsed time: 0' 0\""
            print string_time + " elapsed angle: 0º"
            print string_time + " capture: 0 ms"

        # Setup scanner
        self.driver.board.lasers_off()
        if self.move_motor:
            self.driver.board.motor_enable()
            self.driver.board.motor_reset_origin()
            self.driver.board.motor_speed(self.motor_speed)
            self.driver.board.motor_acceleration(self.motor_acceleration)
        else:
            self.driver.board.motor_disable()

        self.ph_save_enable = profile.settings['ph_save_enable']
        self.ph_save_folder = profile.settings['ph_save_folder']
        self.ph_save_divider = profile.settings['ph_save_divider']
        if self.ph_save_enable:
            self.ph_save_folder = profile.settings['ph_save_folder'] + datetime.datetime.now().strftime("/scan%Y-%m-%d_%H-%M")
            print self.ph_save_folder
            os.makedirs(self.ph_save_folder)

    def _capture(self):
        # Flush buffer of texture captures
        self.image_capture.flush_laser()
        while self.is_scanning:
            if self._inactive:
                self.image_capture.stream = True
                time.sleep(0.1)
            else:
                self.image_capture.stream = False
                if abs(self._theta) >= 360.0:
                    break
                else:
                    begin = time.time()
                    try:
                        # Capture images
                        capture = self._capture_images()
                        # Put images into queue
                        self._captures_queue.put(capture)
                    except Exception as e:
                        self.is_scanning = False
                        response = (False, e)
                        if self._after_callback is not None:
                            self._after_callback(response)
                        break

                    # Move motor
                    if self.move_motor:
                        self.driver.board.motor_move(self.motor_step)
                    else:
                        time.sleep(0.130)  # Time for 0.45º movement

                    # Update theta
                    self._theta += self.motor_step
                    self._count += 1
                    # Refresh progress
                    if self.motor_step != 0:
                        self._progress = abs(self._theta / self.motor_step)
                        self._range = abs(360.0 / self.motor_step)

                    # Print info
                    self._end = time.time()
                    string_time = str(datetime.datetime.now())[:-3] + " - "

                    if self._debug and system == 'Linux':
                        # Cursor up + remove lines
                        print "\x1b[1A\x1b[1A\x1b[1A\x1b[1A\x1b[2K\x1b[1A"
                        print string_time + " elapsed progress: {0} %".format(
                            int(self._theta / 3.6))
                        print string_time + " elapsed time: {0}".format(
                            time.strftime("%M' %S\"", time.gmtime(self._end - self._begin)))
                        print string_time + " elapsed angle: {0}º".format(
                            float(self._theta))
                        print string_time + " capture: {0} ms".format(
                            int((self._end - begin) * 1000))
            # Sleep
            time.sleep(self._scan_sleep)

        self.driver.board.lasers_off()
        self.driver.board.motor_disable()

    def _capture_images(self):
        capture = ScanCapture()
        capture.theta = np.deg2rad(self._theta)
        capture.count = self._count

        capture.texture = None
        if self.capture_texture:
            capture.texture = self.image_capture.capture_texture()
            # Flush buffer to improve the synchronization when
            # the texture exposure is around 33 ms
            self.image_capture.flush_laser()
        '''
        else:
            r, g, b = self.color
            ones = np.zeros((self.calibration_data.height,
                             self.calibration_data.width, 3), np.uint8)
            ones[:, :, 0] = r
            ones[:, :, 1] = g
            ones[:, :, 2] = b
            capture.texture = ones
        '''
        if self.laser[0] and self.laser[1]:
            capture.lasers = self.image_capture.capture_lasers()
        else:
            for i in xrange(2):
                if self.laser[i]:
                    capture.lasers[i],capture.lasers[2] = self.image_capture.capture_laser(i)

        # Set current video images
        self.current_video.set_texture(capture.texture)
        self.current_video.set_laser(capture.lasers)

        return capture

    def _process(self):
        ret = False
        while self.is_scanning:
            if self._inactive:
                self.image_detection.stream = True
                time.sleep(0.1)
            else:
                self.image_detection.stream = False
                if self._theta >= 360.0:
                    self.is_scanning = False
                    ret = True
                    break
                else:
                    if not self._captures_queue.empty():
                        # Get capture from queue
                        capture = self._captures_queue.get(timeout=0.1)
                        self._captures_queue.task_done()
                        # Process capture
                        self._process_capture(capture)
            # Sleep
            time.sleep(self._scan_sleep)

        if ret:
            response = (True, None)
        else:
            response = (False, ScanError())

        # Cursor down
        # if self._debug and system == 'Linux':
        #     print "\x1b[1C"

        self.image_capture.stream = True

        progress = 0
        if self._range > 0:
            progress = int(100 * self._progress / self._range)
        logger.info("Finish scan {0} %  Time {1}".format(
            progress,
            time.strftime("%M' %S\"", time.gmtime(self._end - self._begin))))

        if self._after_callback is not None:
            self._after_callback(response)

    def _process_capture(self, capture):
        # Current video arrays
        image = None
        images = [None, None]
        points = [None, None]

        # begin = time.time()
#        u_offset = 2.8
#        u_offset = 1.8
#        u_offset = 0.0
        for i in xrange(2):
            if capture.lasers[i] is not None:
                image = capture.lasers[i]
                self.image = image
                # Compute 2D points from images
                points_2d, image = self.laser_segmentation.compute_2d_points(image)

                images[i] = image
                points[i] = points_2d

                # Compute point cloud texture
                u, v = points_2d

                if self.capture_texture:
                    texture = capture.texture[v, np.around(u).astype(int)].T

                else:
                    if self._bicolor:
                        if i == 0:
                            r, g, b = 255, 0, 0 # red laser left
                        else:
                            r, g, b = 0, 255, 255 # cyan
                    else:
                        r, g, b = self.color

                    texture = np.zeros((3, len(v)), np.uint8)
                    texture[0, :] = r
                    texture[1, :] = g
                    texture[2, :] = b


                # Compute point cloud from 2D points
                #points_2d[0][:] += u_offset

                #points_2d = self.point_cloud_generation.undistort_points(points_2d)
                point_cloud = self.point_cloud_generation.compute_point_cloud(
                    capture.theta, points_2d, i)

                if self.point_cloud_callback:
                    self.point_cloud_callback(self._range, self._progress,
                                              (point_cloud, texture, i))

        # Photogrammetry
        if self.ph_save_enable and capture.count % self.ph_save_divider == 0:
            filename = self.ph_save_folder + "/img{:03.03f}.png".format(np.rad2deg(capture.theta))
            #print filename
            if capture.texture is not None:
                self.driver.camera.save_image(filename, capture.texture)
            elif capture.lasers[2] is not None:
                self.driver.camera.save_image(filename, capture.lasers[2])

        # Set current video images
        self.current_video.set_gray(images)
        self.current_video.set_line(points, image)

        # Print info
        """if self._debug and system == 'Linux':
            print string_time + " process: {0} ms".format(
                int((time.time() - begin) * 1000))"""
