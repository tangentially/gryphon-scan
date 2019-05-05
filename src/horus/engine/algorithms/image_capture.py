# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np

from horus.util import profile

from horus import Singleton
from horus.engine.driver.driver import Driver
from horus.engine.calibration.calibration_data import CalibrationData

import logging
logger = logging.getLogger(__name__)

class CameraSettings(object):

    def __init__(self):
        self.driver = Driver()

        self.selected = False
        self.brightness = 0
        self.contrast = 0
        self.saturation = 0
        self.exposure = 0
        # Photo lamps
        self.light = [0,0]
        # Laser capture background
        self.laser_bg = [None, None]
        self.laser_bg_enable = False

    def set_brightness(self, value):
        self.brightness = value
        if self.selected:
            self.driver.camera.set_brightness(value)

    def set_contrast(self, value):
        self.contrast = value
        if self.selected:
            self.driver.camera.set_contrast(value)

    def set_saturation(self, value):
        self.saturation = value
        if self.selected:
            self.driver.camera.set_saturation(value)

    def set_exposure(self, value):
        self.exposure = value
        if self.selected:
            self.driver.camera.set_exposure(value)

    def set_light(self, idx, value):
        try:
            self.light[idx] = value
            if self.selected:
                self.driver.camera.set_light(idx,value)
        except IndexError:
            pass

    def send_all_settings(self):
        self.driver.camera.set_brightness(self.brightness)
        self.driver.camera.set_contrast(self.contrast)
        self.driver.camera.set_saturation(self.saturation)
        self.driver.camera.set_exposure(self.exposure)
        for idx,br in enumerate(self.light):
            self.driver.camera.set_light(idx,br)

    def read_profile(self, mode):
        self.set_brightness(profile.settings['brightness_'+mode])
        self.set_contrast(profile.settings['contrast_'+mode])
        self.set_saturation(profile.settings['saturation_'+mode])
        self.set_exposure(profile.settings['exposure_'+mode])
        self.set_light(0,profile.settings['light1_'+mode])
        self.set_light(1,profile.settings['light2_'+mode])
        if mode == 'laser_calibration':
            self.laser_bg = profile.laser_bg_calibration
            self.laser_bg_enable = profile.laser_bg_calibration_enable
        if mode == 'laser_scanning':
            self.laser_bg = profile.laser_bg_scanning
            self.laser_bg_enable = profile.laser_bg_scanning_enable
        else:
            self.laser_bg = [None, None]
            self.laser_bg_enable = False
            logger.info("Capture profile load: "+mode)

@Singleton
class ImageCapture(object):

    def __init__(self):
        self.driver = Driver()
        self.calibration_data = CalibrationData()

        self.texture_mode = CameraSettings()
        self.laser_mode = CameraSettings()
        self.pattern_mode = CameraSettings()

        self.stream = True
        self._mode = self.pattern_mode
        self._mode.selected = True
        self._remove_background = True
        self._updating = False

    def initialize(self):
        self.texture_mode.initialize()
        self.laser_mode.initialize()
        self.pattern_mode.initialize()

    def set_flush_values(self, texture, laser, pattern, mode):
        self._flush_texture = texture
        self._flush_laser = laser
        self._flush_pattern = pattern
        self._flush_mode = mode

    def set_flush_stream_values(self, texture, laser, pattern, mode):
        self._flush_stream_texture = texture
        self._flush_stream_laser = laser
        self._flush_stream_pattern = pattern
        self._flush_stream_mode = mode

    def set_remove_background(self, value):
        self._remove_background = value

    def set_mode(self, mode):
        if self._mode is not mode:
            self._updating = True
            self._mode.selected = False
            self._mode = mode
            self._mode.selected = True
            self._mode.send_all_settings()
            # wait for camera to adjust to new settings
            if self.stream:
                self.capture_image(flush=self._flush_stream_mode)
            else:
                self.capture_image(flush=self._flush_mode)
            self._updating = False

    def set_mode_texture(self):
        self.set_mode(self.texture_mode)

    def set_mode_laser(self):
        self.set_mode(self.laser_mode)

    def set_mode_pattern(self):
        self.set_mode(self.pattern_mode)

    def flush_texture(self, value=0):
        self.set_mode_texture()
        self.capture_image(flush=value)

    def flush_laser(self, value=0):
        self.set_mode_laser()
        self.capture_image(flush=value)

    def flush_pattern(self, value=0):
        self.set_mode_pattern()
        self.capture_image(flush=value)

    def capture_texture(self):
        self.set_mode(self.texture_mode)
        if self.stream:
            flush = self._flush_stream_texture
        else:
            flush = self._flush_texture
        image = self.capture_image(flush=flush)
        return image

    def _capture_laser(self, index):
        self.set_mode(self.laser_mode)
        self.driver.board.lasers_off()
        self.driver.board.laser_on(index)
        if self.stream:
            flush = self._flush_stream_laser
        else:
            flush = self._flush_laser
        image = self.capture_image(flush=flush)
        self.driver.board.laser_off(index)
        # substract environment laser lines 
        if self._mode.laser_bg_enable:
            try:
                if self._mode.laser_bg[index] is not None and \
                   image is not None:
                    image = cv2.subtract(image, self._mode.laser_bg[index])
            except:
                logger.info('WARNING: Error applying laser BG @ image_capture._capture_laser')
        return image

    def capture_laser(self, index):
        # Capture background
        image_background = None
        self.set_mode(self.laser_mode)
        if self._remove_background:
            self.driver.board.lasers_off()
            if self.stream:
                flush = self._flush_stream_laser
            else:
                flush = self._flush_laser
            image_background = self.capture_image(flush=flush)
        # Capture laser
        image = self._capture_laser(index)
        if image_background is not None:
            if image is not None:
                image = cv2.subtract(image, image_background)
        return [image, image_background]

    def capture_lasers(self):
        # Capture background
        image_background = None
        self.set_mode(self.laser_mode)
        if self._remove_background:
            self.driver.board.lasers_off()
            if self.stream:
                flush = self._flush_stream_laser
            else:
                flush = self._flush_laser
            image_background = self.capture_image(flush=flush)
        # Capture lasers
        images = [None, None, image_background]
        images[0] = self._capture_laser(0)
        images[1] = self._capture_laser(1)

        self.remove_background_subtract(images)
        # test hsV based BG removal
        #if self._mode.light[1] > 2:
        #    self.remove_background_subtract(images)
        #else:
        #    self.remove_background_hsv(images,self._mode.light[1])
        return images

    def capture_all_lasers(self):
        image_background = None
        self.set_mode(self.laser_mode)
        if self.stream:
            flush = self._flush_stream_laser
        else:
            flush = self._flush_laser
        if self._remove_background:
            self.driver.board.lasers_off()
            image_background = self.capture_image(flush=flush)
        self.driver.board.lasers_on()
        image = self.capture_image(flush=flush)
        self.driver.board.lasers_off()

        self.remove_background_subtract([image,image_background])
        # test hsV based BG removal
        #if self._mode.light[1] > 2:
        #    self.remove_background_subtract([image,image_background])
        #else:
        #    self.remove_background_hsv([image,image_background],self._mode.light[1])

        # substract environment laser lines
        if self._mode.laser_bg_enable:
            try:
                if image is not None:
                    for bg in self._mode.laser_bg:
                        if bg is not None:
                            image = cv2.subtract(image, bg)
            except:
                logger.info('WARNING: Error applying laser BG @ image_capture.capture_all_lasers')
        return image

    def capture_pattern(self):
        self.set_mode(self.pattern_mode)
        if self.stream:
            flush = self._flush_stream_pattern
        else:
            flush = self._flush_pattern
        image = self.capture_image(flush=flush)
        return image

    def capture_image(self, flush=0):
        image = self.driver.camera.capture_image(flush=flush)
        return image

    def remove_background_subtract(self,images):
        background = images[-1]
        if background is not None:
            for image in images[:-1]:
                if image is not None:
                    cv2.subtract(image, background, image)

    def remove_background_hsv(self,images, ch):
        background = images[-1]
        if background is not None:
            bg_hsv = cv2.cvtColor(background, cv2.COLOR_RGB2HSV)
            bg_hsv = cv2.split(bg_hsv)[ch]#[2]

            #lower = np.array([0,0,self.threshold_value])
            #upper = np.array([255,255,255])
            for image in images[:-1]:
                if image is not None:
                    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    image_hsv = cv2.split(image_hsv)[ch]#[2]
                    image_hsv = cv2.subtract(image_hsv, bg_hsv)
                    #mask = cv2.inRange(image, lower, upper)
                    #image[np.where(mask==0)] = [0,0,0]
                    image[np.where(image_hsv<self._mode.light[0])] = [0,0,0]
