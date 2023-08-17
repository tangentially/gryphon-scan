# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

from horus.util import profile

from horus.gui.engine import driver, calibration_data, image_capture
from horus.gui.util.video_view import VideoView
from horus.gui.workbench.workbench import Workbench
from horus.gui.workbench.control.panels import CameraControl, LaserControl, \
    MotorControl, GcodeControl


class ControlWorkbench(Workbench):

    def __init__(self, parent):
        Workbench.__init__(self, parent, name=_('Control workbench'))

    def add_panels(self):
        self.add_panel('camera_control', CameraControl)
        self.add_panel('laser_control', LaserControl)
        # self.add_panel('ldr_value', LDRControl)
        self.add_panel('motor_control', MotorControl)
        self.add_panel('gcode_control', GcodeControl)

    def add_pages(self):
        self.add_page('video_view', VideoView(self, self._video_frame))
        self.panels_collection.expandable_panels[
            profile.settings['current_panel_control']].on_title_clicked(None)

    @staticmethod
    def _video_frame():
        return image_capture.capture_image()

    def on_open(self):
        self.pages_collection['video_view'].play()

    def on_close(self):
        try:
            driver.board.lasers_off()
            self.pages_collection['video_view'].stop()
            laser_control = self.panels_collection.expandable_panels['laser_control']
            laser_control.get_control('left_button').control.SetValue(False)
            laser_control.get_control('right_button').control.SetValue(False)
        except:
            pass

    def reset(self):
        self.pages_collection['video_view'].reset()

    def setup_engine(self):
        driver.camera.read_profile()

        image_capture.texture_mode.read_profile('control')
        image_capture.set_mode_texture()

        calibration_data.read_profile_camera()
        calibration_data.read_profile_calibration()

        driver.board.motor_speed(profile.settings['motor_speed_control'])
        driver.board.motor_acceleration(profile.settings['motor_acceleration_control'])

