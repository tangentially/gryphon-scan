# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core

from horus.gui.engine import driver, pattern, calibration_data, laser_triangulation, \
    platform_extrinsics, combo_calibration, image_capture
from horus.util import system as sys
from horus.gui.util.custom_panels import ExpandablePanel, Slider, CheckBox, \
    FloatTextBox, FloatTextBoxArray, FloatLabel, FloatLabelArray, Button, \
    IntLabel, IntTextBox, ComboBox


class PatternSettings(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(self, parent, _("Pattern settings"),
                                 selected_callback=on_selected_callback)

    def add_controls(self):
        self.add_control(
            'pattern_rows', Slider, _("Number of corner rows in the pattern"))
        self.add_control(
            'pattern_columns', Slider, _("Number of corner columns in the pattern"))
        self.add_control(
            'pattern_square_width', FloatTextBox, _("Square width in the pattern (mm)"))
        self.add_control(
            'pattern_origin_distance', FloatTextBox,
            _("Minimum distance between the origin of the pattern (bottom-left corner) "
              "and the pattern's base surface (mm)"))

        self.add_control(
            'pattern_border_l', FloatTextBox, _("Border Left (mm)"))
        self.add_control(
            'pattern_border_r', FloatTextBox, _("Border Right (mm)"))
        self.add_control(
            'pattern_border_t', FloatTextBox, _("Border Top (mm)"))
        self.add_control(
            'pattern_border_b', FloatTextBox, _("Border Bottom (mm)"))

    def update_callbacks(self):
        self.update_callback('pattern_rows', lambda v: self._update_rows(v))
        self.update_callback('pattern_columns', lambda v: self._update_columns(v))
        self.update_callback('pattern_square_width', lambda v: self._update_square_width(v))
        self.update_callback('pattern_origin_distance', lambda v: self._update_origin_distance(v))
        self.update_callback('pattern_border_l', lambda v: self._update_border_l(v))
        self.update_callback('pattern_border_r', lambda v: self._update_border_r(v))
        self.update_callback('pattern_border_t', lambda v: self._update_border_t(v))
        self.update_callback('pattern_border_b', lambda v: self._update_border_b(v))

    def _update_rows(self, value):
        pattern.rows = value

    def _update_columns(self, value):
        pattern.columns = value

    def _update_square_width(self, value):
        pattern.square_width = value

    def _update_origin_distance(self, value):
        pattern.origin_distance = value

    def _update_border_l(self, value):
        pattern.border_l = value

    def _update_border_r(self, value):
        pattern.border_r = value

    def _update_border_t(self, value):
        pattern.border_t = value

    def _update_border_b(self, value):
        pattern.border_b = value


class ScannerAutocheck(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(self, parent, _("Scanner autocheck"),
                                 selected_callback=on_selected_callback,
                                 has_undo=False, has_restore=False)


class RotatingPlatform(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(
            self, parent, _("Rotating platform"),
            selected_callback=on_selected_callback, has_undo=False)

    def add_controls(self):
        self.add_control('motor_step_calibration', FloatTextBox,
                         _("Step for laser and platform calibration"))
        self.add_control('motor_speed_calibration', FloatTextBox,
                         _("Speed for laser and platform calibration"))
        self.add_control('motor_acceleration_calibration', FloatTextBox,
                         _("Acceleration for laser and platform calibration"))

        self.add_control('after_calibration_position', ComboBox,
                         _("Platform position on finish"))

    def update_callbacks(self):
        self.update_callback('motor_step_calibration', self._set_step)
        self.update_callback('motor_speed_calibration', self._set_speed)
        self.update_callback('motor_acceleration_calibration', self._set_acceleration)
        self.update_callback('after_calibration_position', lambda v: self._set_after_calibration_position(v))

    def _set_step(self, value):
        laser_triangulation.motor_step = value
        platform_extrinsics.motor_step = value
        combo_calibration.motor_step = value

    def _set_speed(self, value):
        laser_triangulation.motor_speed = value
        platform_extrinsics.motor_speed = value
        combo_calibration.motor_speed = value

    def _set_acceleration(self, value):
        laser_triangulation.motor_acceleration = value
        platform_extrinsics.motor_acceleration = value
        combo_calibration.motor_acceleration = value

    def _set_after_calibration_position(self, value):
        laser_triangulation.final_move = value
        platform_extrinsics.final_move = value
        combo_calibration.final_move = value

class LaserTriangulation(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(self, parent, _("Laser triangulation"),
                                 selected_callback=on_selected_callback, has_undo=False)

    def add_controls(self):
        self.add_control('distance_left', FloatLabel)
        self.add_control('normal_left', FloatLabelArray)
        self.add_control('distance_right', FloatLabel)
        self.add_control('normal_right', FloatLabelArray)


class PlatformExtrinsics(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(self, parent, _("Platform extrinsics"),
                                 selected_callback=on_selected_callback, has_undo=False)

    def add_controls(self):
        self.add_control('rotation_matrix', FloatTextBoxArray)
        self.add_control('translation_vector', FloatTextBoxArray)


class VideoSettings(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(self, parent, _("Video settings"),
                                 selected_callback=on_selected_callback,
                                 has_undo=False, restore_callback=self._set_resolution)

    def add_controls(self):
        self.add_control('camera_rotate', CheckBox, _("Rotate image"))
        self.add_control('camera_hflip', CheckBox, _("Horizontal flip"))
        self.add_control('camera_vflip', CheckBox, _("Vertical flip"))
        if driver.camera.set_resolution_supported():
            self.add_control('auto_resolution', CheckBox, _("MAX resolution"))
            self.add_control('camera_width', IntTextBox, _("Width"))
            self.add_control('camera_height', IntTextBox, _("Height"))
            self.add_control('set_resolution_button', Button, _("Set resolution"))

            if self.get_control('camera_width').GetValue()<0 or self.get_control('camera_height').GetValue()<0:
                self.get_control('auto_resolution').SetValue(True)
                self._auto_resolution(True)
            
        if driver.camera.focus_supported():
            self.add_control(
                'camera_focus', Slider, _("Manual focus"))

    def update_callbacks(self):
        self.update_callback('camera_rotate', lambda v: driver.camera.set_rotate(v))
        self.update_callback('camera_hflip', lambda v: driver.camera.set_hflip(v))
        self.update_callback('camera_vflip', lambda v: driver.camera.set_vflip(v))
        if driver.camera.set_resolution_supported():
            self.update_callback('auto_resolution', lambda v: self._auto_resolution(v))
            self.update_callback('set_resolution_button', self._set_resolution)
        if driver.camera.focus_supported():
            self.update_callback('camera_focus', lambda v: driver.camera.set_focus(v))

    def _set_resolution(self):
        if not sys.is_darwin():
            old_width = driver.camera._width
            old_height = driver.camera._height

            new_width = self.get_control('camera_width').GetValue()
            new_height = self.get_control('camera_height').GetValue()
            driver.camera.set_resolution(new_width, new_height)

            real_width = driver.camera._width
            real_height = driver.camera._height

            if real_width != new_width or real_height != new_height:
                dlg = wx.MessageDialog(
                    self,
                    _("Your camera does not accept this resolution.\n"
                      "Do you want to use the nearest values?"),
                    _("Wrong resolution"), wx.YES_NO | wx.ICON_QUESTION)
                result = dlg.ShowModal() == wx.ID_YES
                dlg.Destroy()
                if result:
                    driver.camera.set_resolution(real_width, real_height)
                    self.get_control('camera_width').SetValue(real_width)
                    self.get_control('camera_height').SetValue(real_height)
                else:
                    driver.camera.set_resolution(old_width, old_height)
                    self.get_control('camera_width').SetValue(old_width)
                    self.get_control('camera_height').SetValue(old_height)

    def _auto_resolution(self, value):
        if value:
            driver.camera.set_resolution(-1, -1)
            self.get_control('camera_width').SetValue(-1)
            self.get_control('camera_height').SetValue(-1)
            self.get_control('camera_width').Hide()
            self.get_control('camera_height').Hide()
            self.get_control('set_resolution_button').Hide()
        else:
            if driver.is_connected:
                self.get_control('camera_width').SetValue(driver.camera._width)
                self.get_control('camera_height').SetValue(driver.camera._height)
            self.get_control('camera_width').Show()
            self.get_control('camera_height').Show()
            self.get_control('set_resolution_button').Show()

        if sys.is_wx30():
            self.content.SetSizerAndFit(self.content.vbox)
        if sys.is_windows():
            self.parent.Refresh()
        self.parent.Layout()
        self.Layout()


class CameraIntrinsics(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(self, parent, _("Camera intrinsics"),
                                 selected_callback=on_selected_callback, has_undo=False)

    def add_controls(self):
        self.add_control('camera_matrix', FloatTextBoxArray)
        self.add_control('distortion_vector', FloatTextBoxArray)
    '''
        self.add_control(
            'use_distortion', CheckBox,
            _("This option applies lens distortion correction to the video. "
              "This process slows the video feed from the camera"))
    '''
    def update_callbacks(self):
        self.update_callback('camera_matrix', lambda v: self._update_camera_matrix(v))
        self.update_callback('distortion_vector', lambda v: self._update_distortion_vector(v))
#        self.update_callback('use_distortion', lambda v: image_capture.set_use_distortion(v))

    def _update_camera_matrix(self, value):
        calibration_data.camera_matrix = value

    def _update_distortion_vector(self, value):
        calibration_data.distortion_vector = value
