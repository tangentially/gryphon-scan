# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import time
import queue
import numpy as np
import datetime
import os
import threading

from horus import Singleton
from horus.engine.scan.scan import Scan, ScanError
from horus.engine.scan.scan_capture import ScanCapture
from horus.engine.scan.current_video import CurrentVideo
from horus.engine.calibration.calibration_data import CalibrationData
from horus.util.gryphon_util import decode_color

from horus.util import profile

import logging
logger = logging.getLogger(__name__)

import platform
system = platform.system()


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
        self.texture_mode = 2 # Capture 
        self.laser = [True]*len(self.calibration_data.laser_planes)
        self.move_motor = True
        self.motor_step = 0
        self.motor_speed = 0
        self.motor_acceleration = 0
        self.color = (0, 0, 0)
        self.colors = [(0, 0, 0)]*len(self.calibration_data.laser_planes)

        self._theta = 0
        self._count = 0
        self._debug = False
        self._scan_sleep = 0.05
        self._captures_queue = queue.Queue(10)
        self.point_cloud_callback = None

        self.ph_save_enable = False
        self.ph_save_folder = 'photo/'
        self.ph_save_divider = 1

        self.capturing = False
        self.semaphore = None

    def read_profile(self):
        self.set_texture_mode(profile.settings['texture_mode'])

        self.set_color(profile.settings['point_cloud_color'])
        self.set_colors(0, profile.settings['point_cloud_color_l'])
        self.set_colors(1, profile.settings['point_cloud_color_r'])

        use_laser = profile.settings['use_laser']
        self.set_use_left_laser(use_laser == 'Left' or use_laser == 'Both')
        self.set_use_right_laser(use_laser == 'Right' or use_laser == 'Both')

        self.motor_step = profile.settings['motor_step_scanning']
        self.motor_speed = profile.settings['motor_speed_scanning']
        self.motor_acceleration = profile.settings['motor_acceleration_scanning']

        self.set_scan_sleep(profile.settings['scan_sleep'])
        if profile.settings['scan_sync_threads']:
            self.semaphore = threading.Semaphore()
        else:
            self.semaphore = None


        self.ph_save_enable = profile.settings['ph_save_enable']
        self.ph_save_folder = profile.settings['ph_save_folder']
        self.ph_save_divider = profile.settings['ph_save_divider']

    def set_texture_mode(self, value):
        # 'Flat color', 'Multi color', 'Capture', 'Laser BG'
        if value == 'Flat color':
            self.texture_mode = 0
        elif value == 'Multi color':
            self.texture_mode = 1
        elif value == 'Capture':
            self.texture_mode = 2
        elif value == 'Laser BG':
            self.texture_mode = 3
        else:
            self.texture_mode = 2

    def set_color(self, value):
        self.color = decode_color(value) 

    def set_colors(self, index, value):
        self.colors[index] = decode_color(value) 

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
        self.capturing = False
        self._begin = time.time()

        # Setup console
        logger.info("Start scan")
        if self._debug and system == 'Linux':
            string_time = str(datetime.datetime.now())[:-3] + " - "
            print(string_time + " elapsed progress: 0 %")
            print(string_time + " elapsed time: 0' 0\"")
            print(string_time + " elapsed angle: 0º")
            print(string_time + " capture: 0 ms")

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
            print(self.ph_save_folder)
            os.makedirs(self.ph_save_folder)

    def _capture(self):
        self.capturing = True
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
                        if self.semaphore is not None:
                            self.semaphore.acquire()
                        capture = self._capture_images()
                        if self.semaphore is not None:
                            self.semaphore.release()
                        # Put images into queue
                        self._captures_queue.put(capture)
                    except Exception as e:
                        logger.info("Capture error: "+str(e))
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
                    #self._end = time.time()
                    #print "capture: {0} ms".format(
                    #    int((self._end - begin) * 1000))

                    if self._debug and system == 'Linux':
                        string_time = str(datetime.datetime.now())[:-3] + " - "
                        # Cursor up + remove lines
                        print("\x1b[1A\x1b[1A\x1b[1A\x1b[1A\x1b[2K\x1b[1A")
                        print(string_time + " elapsed progress: {0} %".format(
                            int(self._theta / 3.6)))
                        print(string_time + " elapsed time: {0}".format(
                            time.strftime("%M' %S\"", time.gmtime(self._end - self._begin))))
                        print(string_time + " elapsed angle: {0}º".format(
                            float(self._theta)))
                        print(string_time + " capture: {0} ms".format(
                            int((self._end - begin) * 1000)))
            # Sleep
            time.sleep(self._scan_sleep)

        self.driver.board.lasers_off()
        self.driver.board.motor_disable()
        self.capturing = False
        self.image_capture.stream = True

    def _capture_images(self):
        capture = ScanCapture(lasers = len(self.laser))
        capture.theta = np.deg2rad(self._theta)
        capture.count = self._count

        if self.texture_mode == 2:
            capture.texture = self.image_capture.capture_texture()

        if all(self.laser):
            capture.lasers = self.image_capture.capture_lasers()
        else:
            for i in range(len(self.laser)):
                if self.laser[i]:
                    # TODO Use previous captured background
                    capture.lasers[i],capture.lasers[-1] = self.image_capture.capture_laser(i)

        # Set current video images
        if self.texture_mode == 3:
            self.current_video.set_texture(capture.lasers[-1])
        else:
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
                if not self._captures_queue.empty():
                    # Get capture from queue
                    capture = self._captures_queue.get(timeout=0.1)
                    self._captures_queue.task_done()
                    # Process capture
                    self._process_capture(capture)

                    # if last data processed
                    if capture.theta >= 2*np.pi: # 360.0:
                        print("Final angle processed. Shutdown processing thread.")
                        self.is_scanning = False
                        ret = True
                        break
                else:
                    if self.capturing:
                        # Wait for more data
                        time.sleep(0.1)
                    else:
                        print("No more data expected. Shutdown processing thread.")
                        self.is_scanning = False
                        ret = True
                        break

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

        self._end = time.time()
        logger.info("Finish scan {0} %  Time {1}".format(
            progress,
            time.strftime("%M' %S\"", time.gmtime(self._end - self._begin))))

        if self._after_callback is not None:
            self._after_callback(response)


    def _process_capture(self, capture):
        # Current video arrays
        image = None
        points = [None, None]

        #print("Process start: {0:f}".format(np.rad2deg(capture.theta)))
        # begin = time.time()
        for i in range(2):
            if capture.lasers[i] is not None:
                #print "Process image {0} at angle {1}".format(i,np.rad2deg(capture.theta))
                if self.semaphore is not None:
                    self.semaphore.acquire()
                image = capture.lasers[i]
                self.image = image
                # Compute 2D points from images
                points_2d, image = self.laser_segmentation.compute_2d_points(image)
                points[i] = points_2d

                # Compute point cloud texture
                u, v = points_2d
                texture = None
                if self.texture_mode == 1:
                    # Multi color
                    r, g, b = self.colors[i]
                    texture = np.zeros((3, len(v)), np.uint8)
                    texture[0, :] = r
                    texture[1, :] = g
                    texture[2, :] = b

                elif self.texture_mode == 2:
                    # Texture
                    if capture.texture is not None:
                        texture = capture.texture[v, np.around(u).astype(int)].T

                elif self.texture_mode == 3:
                    # Laser BG
                    if capture.lasers[-1] is not None:
                        texture = capture.lasers[-1][v, np.around(u).astype(int)].T

                if texture is None:
                    # Flat color fallback
                    r, g, b = self.color
                    texture = np.zeros((3, len(v)), np.uint8)
                    texture[0, :] = r
                    texture[1, :] = g
                    texture[2, :] = b

                point_cloud = self.point_cloud_generation.compute_point_cloud(
                    capture.theta, points_2d, i)
                #print("Processed: {0:f} - {1}".format(np.rad2deg(capture.theta),i))

                if self.point_cloud_callback:
                    self.point_cloud_callback(self._range, self._progress,
                                              (point_cloud, texture), (i, capture.count, capture.theta))

                if self.semaphore is not None:
                    self.semaphore.release()
        # Photogrammetry
        if self.semaphore is not None:
            self.semaphore.acquire()
        if self.ph_save_enable and capture.count % self.ph_save_divider == 0:
            filename = self.ph_save_folder + "/img{:03.03f}.png".format(np.rad2deg(capture.theta))
            #print filename
            if capture.texture is not None:
                self.driver.camera.save_image(filename, capture.texture)
            elif capture.lasers[-1] is not None:
                self.driver.camera.save_image(filename, capture.lasers[-1])

        # Set current video images
        self.current_video.set_gray(capture.lasers[:-1])
        self.current_video.set_line(points, capture.lasers[-1])
        if self.semaphore is not None:
            self.semaphore.release()

        # Print info
        #print("Process end: {0:f} {1}ms".format(np.rad2deg(capture.theta), int((time.time() - begin) * 1000) ))
