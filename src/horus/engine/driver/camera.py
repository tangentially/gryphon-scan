# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'


import logging
logger = logging.getLogger(__name__)

class WrongCamera(Exception):

    def __init__(self):
        Exception.__init__(self, "Wrong Camera")


class CameraNotConnected(Exception):

    def __init__(self):
        Exception.__init__(self, "Camera Not Connected")


class InvalidVideo(Exception):

    def __init__(self):
        Exception.__init__(self, "Invalid Video")


class WrongDriver(Exception):

    def __init__(self):
        Exception.__init__(self, "Wrong Driver")


class InputOutputError(Exception):

    def __init__(self):
        Exception.__init__(self, "V4L2 Input/Output Error")


class Camera(object):

    """Camera class. For accessing to the scanner camera"""

    def __init__(self, parent=None, camera_id=0):
        self.parent = parent
        self.camera_id = camera_id
        self.unplug_callback = None

        self.initialize()


    def initialize(self):
        self._brightness = 0
        self._contrast = 0
        self._saturation = 0
        self._exposure = 0
        self._luminosity = 1.0
        self._frame_rate = 0
        self._width = 0
        self._height = 0
        self._rotate = True
        self._hflip = True
        self._vflip = False

    def connect(self):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def set_unplug_callback(self, value):
        self.unplug_callback = value

    def capture_image(self, flush=0, auto=False):
        raise NotImplementedError

    def save_image(self, filename, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image)

    def set_rotate(self, value):
        self._rotate = value

    def set_hflip(self, value):
        self._hflip = value

    def set_vflip(self, value):
        self._vflip = value

    def set_brightness(self, value):
        self._brightness = value

    def set_contrast(self, value):
        self._contrast = value

    def set_saturation(self, value):
        self._saturation = value

    def set_exposure(self, value, force=False):
        self._exposure = value

    def set_luminosity(self, value):
        possible_values = {
            "High": 0.5,
            "Medium": 1.0,
            "Low": 2.0
        }
        self._luminosity = possible_values[value]

    def set_frame_rate(self, value):
        self._frame_rate = value

    def set_resolution(self, width, height):
        self._width = width
        self._height = height

    def set_light(self, idx, brightness):
        raise NotImplementedError

    def get_brightness(self):
        return self._brightness

    def get_exposure(self):
        return self._exposure

    def get_resolution(self):
        if self._rotate:
            return int(self._height), int(self._width)
        else:
            return int(self._width), int(self._height)

    def get_video_list(self):
        raise NotImplementedError
