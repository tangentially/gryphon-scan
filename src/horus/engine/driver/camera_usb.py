# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>, Mikhail Klimushin <gryphon@night-gryphon.ru>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L., Copyright (C) 2018-2019 Mikhail Klimushin'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

from horus.util import profile

import cv2
import math
import time
import glob
import platform
import wx

from horus.engine.driver.camera import Camera, WrongCamera, CameraNotConnected, InvalidVideo, \
    WrongDriver

from distutils.version import StrictVersion, LooseVersion

import logging
logger = logging.getLogger(__name__)

system = platform.system()

if system == 'Darwin':
    import uvc
    from uvc.mac import *


class Camera_usb(Camera):

    """Camera class. For accessing to the scanner camera"""

    def __init__(self, parent=None, camera_id=0):
        Camera.__init__(self)

        self._capture = None
        self.controls = None
        self._is_connected = False
        self._reading = False
        self._updating = False
        self._last_image = None
        self._video_list = None
        self._tries = 0  # Check if command fails

        self._auto_resolution = False

        self.initialize()

        self._max_exposure = 1. # fallback value

        if system == 'Windows':
            self._number_frames_fail = 3
            self._max_brightness = 255. #1.
            self._max_contrast   = 255. #1.
            self._max_saturation = 255. #1.
        elif system == 'Darwin':
            self._number_frames_fail = 3
            self._max_brightness = 255.
            self._max_contrast = 255.
            self._max_saturation = 255.
            self._rel_exposure = 10.
        elif system == 'Linux':
            self._number_frames_fail = 3
            # For v4l, these values are scaled from [0, 1) and
            # should thus match the maximum values shown in the
            # UI (see util/profile.py)
            self._max_brightness = 255.0
            self._max_contrast = 255.0
            self._max_saturation = 255.0
            self._max_exposure = 64.0
        else:
            self._number_frames_fail = 3
            self._max_brightness = 255.
            self._max_contrast = 255.
            self._max_saturation = 255.
            self._max_exposure = 1000.

        if LooseVersion(cv2.__version__) > LooseVersion("3.0.0"):
            self.CV_CAP_PROP_BRIGHTNESS   = cv2.CAP_PROP_BRIGHTNESS
            self.CV_CAP_PROP_CONTRAST     = cv2.CAP_PROP_CONTRAST
            self.CV_CAP_PROP_SATURATION   = cv2.CAP_PROP_SATURATION
            self.CV_CAP_PROP_EXPOSURE     = cv2.CAP_PROP_EXPOSURE
            self.CV_CAP_PROP_FPS          = cv2.CAP_PROP_FPS
            self.CV_CAP_PROP_FRAME_WIDTH  = cv2.CAP_PROP_FRAME_WIDTH
            self.CV_CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
            self.CV_CAP_PROP_AUTO_EXPOSURE = cv2.CAP_PROP_AUTO_EXPOSURE
        else:
            self.CV_CAP_PROP_BRIGHTNESS   = cv2.cv.CV_CAP_PROP_BRIGHTNESS
            self.CV_CAP_PROP_CONTRAST     = cv2.cv.CV_CAP_PROP_CONTRAST
            self.CV_CAP_PROP_SATURATION   = cv2.cv.CV_CAP_PROP_SATURATION
            self.CV_CAP_PROP_EXPOSURE     = cv2.cv.CV_CAP_PROP_EXPOSURE
            self.CV_CAP_PROP_FPS          = cv2.cv.CV_CAP_PROP_FPS
            self.CV_CAP_PROP_FRAME_WIDTH  = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
            self.CV_CAP_PROP_FRAME_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
            self.CV_CAP_PROP_AUTO_EXPOSURE = cv2.cv.CV_CAP_PROP_AUTO_EXPOSURE


    def connect(self):
        logger.info("Connecting camera {0}".format(self.camera_id))
        self._is_connected = False
        self.initialize()
        if system == 'Darwin':
            # try to get USB camera params control intrface
            logger.info("Connecting USB Video controls")
            self.controls = None
            try:
                for device in uvc.mac.Camera_List():
                    if device.src_id == self.camera_id:
                        self.controls = uvc.mac.Controls(device.uId)
            except:
                self.controls = None
                wx.MessageDialog(None, 'For MacOS this camera controls not available. You can not set Brightness, Contrast, Saturation, Exposure for this camera', 'Warning', wx.OK | wx.ICON_INFORMATION).ShowModal()

        if self._capture is not None:
            self._capture.release()

        if system == 'Windows':
            if LooseVersion(cv2.__version__) > LooseVersion("3.4.4"):
                self._capture = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            else:
                self._capture = cv2.VideoCapture(self.camera_id)
        else:
            #self._capture = cv2.VideoCapture(self.camera_id, cv2.CAP_FFMPEG)
            self._capture = cv2.VideoCapture(self.camera_id)

        time.sleep(0.2)
        if not self._capture.isOpened():
            time.sleep(1)
            if system == 'Windows':
                if LooseVersion(cv2.__version__) > LooseVersion("3.4.4"):
                    self._capture.open(self.camera_id, cv2.CAP_DSHOW)
                else:
                    self._capture.open(self.camera_id)
            else:
                #self._capture.open(self.camera_id, cv2.CAP_FFMPEG)
                self._capture.open(self.camera_id)

        if self._capture.isOpened():
            self._is_connected = True

            if profile.settings['camera_capture_before_set']:
                self._check_video()

            # set initial resolution to auto, FPS to 30
            # assume BEFORE any frames captured set resolution and FPS applicable for all backends (?)
            logger.info("  set initial resolution/FPS")
            self._auto_resolution = False
            self.set_resolution(
                profile.settings['camera_width'], profile.settings['camera_height'], True)
            self.set_frame_rate(int(profile.settings['frame_rate']), True)

            # disable Auto Exposure
            if LooseVersion(cv2.__version__) > LooseVersion("3.0.0"):
                self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                self.get_exposure()

            if system == 'Darwin' and self.controls is not None:
                self.controls['UVCC_REQ_EXPOSURE_AUTOMODE'].set_val(1)

            # disable Auto Focus
            if LooseVersion(cv2.__version__) > LooseVersion("3.4.4"):
                self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self.get_focus()

            if system == 'Darwin' and self.controls is not None:
                self.controls['UVCC_REQ_FOCUS_AUTO'].set_val(0)

            # Anti flicker
            self.set_anti_flicker(1)

            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            logger.info("  check read frame")
            self._check_video()

            logger.info("  check adjust exposure/brightness")
            self.DetectLimits()
            self._check_camera()

            #logger.info("  check win driver bug")
            #self._check_driver()

            logger.info(" Done")
        else:
            raise CameraNotConnected()

    def disconnect(self):
        tries = 0
        if self._is_connected:
            logger.info("Disconnecting camera {0}".format(self.camera_id))
            if self._capture is not None:
                if self._capture.isOpened():
                    self._is_connected = False
                    while tries < 10:
                        tries += 1
                        if not self._reading:
                            self._capture.release()
                            cv2.destroyAllWindows()
                logger.info(" Done")

    def _check_video(self):
        """Check correct video"""
        frame = self.capture_image(flush=1)
        if frame is None or (frame == 0).all():
            raise InvalidVideo()

    def _check_camera(self):
        # skip check for unsupported system
        if system == 'Darwin' and self.controls is None:
            logger.info("  [Skip unsupported]")
            return True

        """Check correct camera"""
        c_exp = False
        c_bri = False

        try:
            # Check exposure
            self.set_exposure(2)
            exposure = self.get_exposure()
            if exposure is not None:
                c_exp = exposure >= 1.9

            # Check brightness
            self.set_brightness(128) #2)
            brightness = self.get_brightness()
            if brightness is not None:
                c_bri = brightness > 0 #>= 2
        except:
            raise WrongCamera()

        if not c_exp or not c_bri:
            raise WrongCamera()

    def _check_driver(self):
        """Check correct driver: only for Windows"""
        if system == 'Windows':
            self.set_exposure(10)
            frame = self.capture_image(flush=1)
            mean = sum(cv2.mean(frame)) / 3.0
            if mean > 200:
                raise WrongDriver()

    def capture_image(self, flush=0):
        """Capture image from camera"""
        # flush buffered frames
        # 0 - no flush
        # -1 - auto flush
        # n - flush exactly n frames
        if self._is_connected:
            #tbegin = time.time()
            if self._updating:
                return self._last_image
            else:
                self._reading = True
                # Note: Windows needs read() to perform
                #       the flush instead of grab()
                if flush < 0:
                    b, e = 0, 0
                    c = 0
                    # max time for buffered frame
                    # max flushed frames count
                    while b - e > (flush * 0.001) and c < 4:
                        b = time.time()
                        #self._capture.grab()
                        ret, image = self._capture.read()
                        e = time.time()
                        c += 1
                        #print "     frame {1}: {0} ms".format(int((e - b) * 1000), c)
                else:
                    for i in xrange(flush+1):
                        #b = time.time()
                        ret, image = self._capture.read()
                        #e = time.time()
                        #print "     frame: {0} ms".format(int((e - b) * 1000))

                #print "   driver capture: {0} ms, flush {1}".format(int((time.time() - tbegin) * 1000),flush)
                #tbegin = time.time()

                self._reading = False
                if ret:
                    if self._rotate:
                        image = cv2.transpose(image)
                    if self._hflip:
                        image = cv2.flip(image, 1)
                    if self._vflip:
                        image = cv2.flip(image, 0)
                    self._success()
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self._last_image = image
                    #print "   driver capture process: {0} ms".format(int((time.time() - tbegin) * 1000))
                    return image
                else:
                    self._fail()
                    return None
        else:
            return None

    # ------------- Probe limits ----------
    def DetectPropMax(self, min, max, prop_id):
        if prop_id is None:
            return
        while min < max-1:
            v=int((min+max)/2)
            ret = self._capture.set(prop_id, v)
            if ret:
                min = v
            else:
                max = v
        return min

    def DetectLimits(self):
        if not self._is_connected:
            return

        if system == 'Windows':
            self._max_brightness = self.DetectPropMax(0, 255, self.CV_CAP_PROP_BRIGHTNESS)
            self._max_contrast   = self.DetectPropMax(0, 255, self.CV_CAP_PROP_CONTRAST)
            self._max_exposure   = self.DetectPropMax(-255, 255, self.CV_CAP_PROP_EXPOSURE)
            self._max_saturation = self.DetectPropMax(0, 255, self.CV_CAP_PROP_SATURATION)
            print "Max Bri {0} Contr {1} Exp {2} Sat {3}".format(
                self._max_brightness, self._max_contrast,
                self._max_exposure, self._max_saturation)

    # ------------- Brightness control ------------
    def get_brightness(self):
        if self._is_connected:
            if system == 'Darwin' and self.controls is not None:
                ctl = self.controls['UVCC_REQ_BRIGHTNESS_ABS']
                value = ctl.get_val()
            else:
                value = self._capture.get(self.CV_CAP_PROP_BRIGHTNESS)
                #value = value * self._max_brightness
                value = value * 255. / self._max_brightness
            return value

    def set_brightness(self, value):
        if self._is_connected:
            if self._brightness != value:
                self._updating = True
                self._brightness = value
                if system == 'Darwin' and self.controls is not None:
                    ctl = self.controls['UVCC_REQ_BRIGHTNESS_ABS']
                    ctl.set_val(self._line(value, 0, self._max_brightness, ctl.min, ctl.max))
                else:
                    #value = int(value) / self._max_brightness
                    value = value * self._max_brightness / 255.
                    ret = self._capture.set(self.CV_CAP_PROP_BRIGHTNESS, value)
                self._updating = False
                return True
        return False

    # ------------- Contrast control ------------
    def set_contrast(self, value):
        if self._is_connected:
            if self._contrast != value:
                self._updating = True
                self._contrast = value
                if system == 'Darwin' and self.controls is not None:
                    ctl = self.controls['UVCC_REQ_CONTRAST_ABS']
                    ctl.set_val(self._line(value, 0, self._max_contrast, ctl.min, ctl.max))
                else:
                    value = value * self._max_contrast / 255.
                    ret = self._capture.set(self.CV_CAP_PROP_CONTRAST, value)
                self._updating = False
                return True
        return False

    # ------------- Saturation control ------------
    def set_saturation(self, value):
        if self._is_connected:
            if self._saturation != value:
                self._updating = True
                self._saturation = value
                if system == 'Darwin' and self.controls is not None:
                    ctl = self.controls['UVCC_REQ_SATURATION_ABS']
                    ctl.set_val(self._line(value, 0, self._max_saturation, ctl.min, ctl.max))
                else:
                    value = value * self._max_saturation / 255.
                    ret = self._capture.set(self.CV_CAP_PROP_SATURATION, value)
                    if system == 'Windows' and not ret:
                        print "ERROR Set Exposure {0}".format(value)
                    self._updating = False
                return True
        return False

    # ------------- Exposure control ------------
    def get_exposure(self):
        if self._is_connected:
            if system == 'Darwin' and self.controls is not None:
                ctl = self.controls['UVCC_REQ_EXPOSURE_ABS']
                value = ctl.get_val()
                value /= self._rel_exposure
            elif system == 'Windows':
                value = self._capture.get(self.CV_CAP_PROP_EXPOSURE)
                value = 2 ** -value
                #value = value / self._max_exposure * 64
            else:
                value = self._capture.get(self.CV_CAP_PROP_EXPOSURE)
                value *= self._max_exposure
            return value

    def set_exposure(self, value, force=False):
        if self._is_connected:
            if self._exposure != value or force:
                self._updating = True
                self._exposure = value
                #value *= self._luminosity
                if value < 1:
                    value = 1
                if system == 'Darwin' and self.controls is not None:
                    ctl = self.controls['UVCC_REQ_EXPOSURE_ABS']
                    value = int(value * self._rel_exposure)
                    ctl.set_val(value)

                    self.set_anti_flicker(1)
                elif system == 'Windows':
                    value = int(round(-math.log(value) / math.log(2)))
                    #value = value / 64 * self._max_exposure
                    self._capture.set(self.CV_CAP_PROP_EXPOSURE, value)
                else:
                    value = int(value) / self._max_exposure * 5000
                    ret = self._capture.set(self.CV_CAP_PROP_EXPOSURE, value)
                self._updating = False
                return True
        return False

    def set_luminosity(self, value):
        Camera.set_luminosity(self, value)
        return self.set_exposure(self._exposure, force=True)

    # ------------- Anti flicker ------------
    def set_anti_flicker(self, value):
        if system == 'Darwin' and self.controls is not None:
            #self.capture_image(3)
            ctl = self.controls['UVCC_REQ_POWER_LINE_FREQ']
            #print "Line Freq {0} ; {1}".format(ctl.min, ctl.max)
            ctl.set_val(0)
            ctl.set_val(value)

    # ------------- Frame rate control ------------
    def set_frame_rate(self, value, init_phase=False):
	logger.info("Set Frame rate: {0}".format(value))
        if self._is_connected:
            if not init_phase:
                if system == 'Windows':
                    if LooseVersion(cv2.__version__) >= LooseVersion("3.4.4"):
                        if self._capture.getBackendName() in ["MSMF"]:
                            logger.info("UNSUPPORTED for this video backend {0}".format(self._capture.getBackendName()))
                            return
                    else:
                        logger.info("Possible unsupported. Skipping.")
                        return

            if self._frame_rate != value:
                self._frame_rate = value
                self._updating = True
                self._capture.set(self.CV_CAP_PROP_FPS, value)
                self._updating = False

    # ------------- Resolution control ------------
    def set_resolution_supported(self, init_phase=False):
        # init_phase - bypass for systems support set resolution before capture first frame
        if system == 'Darwin':
            return False

        if self._is_connected and not init_phase:
            if LooseVersion(cv2.__version__) >= LooseVersion("3.4.4"):
                if self._capture.getBackendName() in ["MSMF"]:
                    return False
            else:
                if system == 'Windows':
                    return False
        return True

    def get_resolution(self):
        if self._rotate:
            return int(self._height), int(self._width)
        else:
            return int(self._width), int(self._height)

    def set_resolution(self, width, height, init_phase = False):
        logger.info("Set Resolution: {0}x{1}".format(width, height))
        if self._is_connected and self.set_resolution_supported(init_phase):
            if width>0 and height>0:
                self._auto_resolution = False
            else:
                if self._auto_resolution:
                    return
                self._auto_resolution = True
                width = 10000
                height = 10000

            if self._width != width or self._height != height:
                self._updating = True
                self._set_width(width)
                self._set_height(height)
                self._update_resolution()
                self._updating = False

    def _set_width(self, value):
        self._capture.set(self.CV_CAP_PROP_FRAME_WIDTH, value)

    def _set_height(self, value):
        self._capture.set(self.CV_CAP_PROP_FRAME_HEIGHT, value)

    def _update_resolution(self):
        self._width = int(self._capture.get(self.CV_CAP_PROP_FRAME_WIDTH))
        self._height = int(self._capture.get(self.CV_CAP_PROP_FRAME_HEIGHT))
        logger.info("Actual Resolution: {0}x{1}".format(self._width, self._height))

    # ------------- Focus control ------------
    def focus_supported(self):
        if system == 'Darwin':
            return (not self._is_connected) or (self.controls is not None)
        else:
            return LooseVersion(cv2.__version__) > LooseVersion("3.4.4")

    def set_focus(self, value):
        if self._is_connected:
           if system == 'Darwin' and self.controls is not None:
               ctl = self.controls['UVCC_REQ_FOCUS_ABS']
               ctl.set_val(self._line(value, 0, self._max_brightness, ctl.min, ctl.max))
           elif LooseVersion(cv2.__version__) > LooseVersion("3.4.4"):
               self._capture.set(cv2.CAP_PROP_FOCUS, value)
        self._focus = value

    def get_focus(self):
        if self._is_connected:
            if system == 'Darwin' and self.controls is not None:
                ctl = self.controls['UVCC_REQ_FOCUS_ABS']
                self._focus = ctl.get_val()
            elif LooseVersion(cv2.__version__) > LooseVersion("3.4.4"):
                self._focus = self._capture.get(cv2.CAP_PROP_FOCUS)
        return self._focus

    # ------------- Photo lights control ------------
    def set_light(self, idx, brightness):
        if self.parent is not None and \
           not self.parent.unplugged and \
           self.parent.board is not None:
            return self.parent.board.set_light(idx,brightness)
        return False

    def _success(self):
        self._tries = 0

    def _fail(self):
        logger.debug("Camera fail")
        self._tries += 1
        if self._tries >= self._number_frames_fail:
            self._tries = 0
            if self.unplug_callback is not None and \
               self.parent is not None and \
               not self.parent.unplugged:
                self.parent.unplugged = True
                self.unplug_callback()

    def _line(self, value, imin, imax, omin, omax):
        ret = 0
        if omin is not None and omax is not None:
            if (imax - imin) != 0:
                ret = int((value - imin) * (omax - omin) / (imax - imin) + omin)
        return ret

    def _count_cameras(self):
        for i in xrange(5):
            cap = cv2.VideoCapture(i)
            res = not cap.isOpened()
            cap.release()
            if res:
                return i
        return 5

    def get_video_list(self):
        baselist = []
        if system == 'Windows':
            if not self._is_connected:
                count = self._count_cameras()
                for i in xrange(count):
                    baselist.append(str(i))
                self._video_list = baselist
        elif system == 'Darwin':
            for device in uvc.mac.Camera_List():
                baselist.append(str(device.src_id))
            self._video_list = baselist
        else:
            for device in ['/dev/video*']:
                baselist = baselist + glob.glob(device)
            self._video_list = baselist
        return self._video_list

