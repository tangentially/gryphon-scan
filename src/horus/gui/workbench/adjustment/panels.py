# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx

from horus.util import profile, system as sys

from horus.gui.engine import image_capture, laser_segmentation

from horus.gui.workbench.adjustment.current_video import CurrentVideo
from horus.gui.util.custom_panels import ExpandablePanel, Slider, ComboBox, CheckBox

current_video = CurrentVideo()


class ScanCapturePanel(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        self.avoid_platform = False
        ExpandablePanel.__init__(self, parent, _("Scan capture"))

    def add_controls(self):
        self.add_control('capture_mode_scanning', ComboBox)
        self.add_control(
            'brightness_texture_scanning', Slider,
            _("Image luminosity. Low values are better for environments with high ambient "
              "light conditions. High values are recommended for poorly lit places"))
        self.add_control(
            'contrast_texture_scanning', Slider,
            _("Relative difference in intensity between an image point and its "
              "surroundings. Low values are recommended for black or very dark colored "
              "objects. High values are better for very light colored objects"))
        self.add_control(
            'saturation_texture_scanning', Slider,
            _("Purity of color. Low values will cause colors to disappear from the image. "
              "High values will show an image with very intense colors"))
        self.add_control(
            'exposure_texture_scanning', Slider,
            _("Amount of light per unit area. It is controlled by the time the camera "
              "sensor is exposed during a frame capture. "
              "High values are recommended for poorly lit places"))

        self.add_control(
            'light1_texture_scanning', Slider,
            _("Photo lamp 1 brightness"))
        self.add_control(
            'light2_texture_scanning', Slider,
            _("Photo lamp 2 brightness"))


        self.add_control(
            'brightness_laser_scanning', Slider,
            _("Image luminosity. Low values are better for environments with high ambient "
              "light conditions. High values are recommended for poorly lit places"))
        self.add_control(
            'contrast_laser_scanning', Slider,
            _("Relative difference in intensity between an image point and its "
              "surroundings. Low values are recommended for black or very dark colored "
              "objects. High values are better for very light colored objects"))
        self.add_control(
            'saturation_laser_scanning', Slider,
            _("Purity of color. Low values will cause colors to disappear from the image. "
              "High values will show an image with very intense colors"))
        self.add_control(
            'exposure_laser_scanning', Slider,
            _("Amount of light per unit area. It is controlled by the time the camera "
              "sensor is exposed during a frame capture. "
              "High values are recommended for poorly lit places"))

        self.add_control(
            'light1_laser_scanning', Slider,
            _("Photo lamp 1 brightness"))
        self.add_control(
            'light2_laser_scanning', Slider,
            _("Photo lamp 2 brightness"))

        self.add_control(
            'remove_background_scanning', CheckBox,
            _("Capture an extra image without laser to remove "
              "the background in the laser's image"))

        # Initial layout
        self._set_mode_layout(profile.settings['capture_mode_scanning'])

        # capture bg buttons panel
        control = wx.Panel(self.content)
        control.SetToolTip(wx.ToolTip("Laser background filter"))

        label = wx.StaticText(control, label=_("Laser background filter"), style=wx.ALIGN_CENTER)
        enable_box = wx.CheckBox(control, label="Enable filter")
        platform_box = wx.CheckBox(control, label="Don't mask platform")
        buttons_box = wx.Panel(control)
        buttons_box.left_button = wx.Button(buttons_box, -1, "Reset")
        buttons_box.right_button = wx.Button(buttons_box, -1, "Capture")

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(label, 0, wx.TOP | wx.EXPAND, border=5)
        vbox.Add(enable_box, 0, wx.TOP | wx.EXPAND, border=5)
        vbox.Add(platform_box, 0, wx.TOP | wx.EXPAND, border=5)
        vbox.Add(buttons_box, 0, wx.TOP | wx.EXPAND, border=5)
        control.SetSizer(vbox)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(buttons_box.left_button, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        hbox.AddStretchSpacer()
        hbox.Add(buttons_box.right_button, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        buttons_box.SetSizer(hbox)

        control.Layout()

        # Events
        buttons_box.left_button.Bind(wx.EVT_BUTTON, lambda v: self.laser_reset() )
        buttons_box.right_button.Bind(wx.EVT_BUTTON, lambda v: self.laser_capture() )
        enable_box.Bind(wx.EVT_CHECKBOX, lambda v: self.laser_bg_enable(v.GetEventObject().GetValue()) )
        platform_box.Bind(wx.EVT_CHECKBOX, lambda v: self.set_avoid_platform(v.GetEventObject().GetValue()) )

        enable_box.SetValue(profile.laser_bg_scanning_enable)
        platform_box.SetValue(self.avoid_platform)

        self.content.vbox.Add(control, 0, wx.BOTTOM | wx.EXPAND, 5)
        self.content.vbox.Layout()
        if sys.is_wx30():
            self.content.SetSizerAndFit(self.content.vbox)


    def update_callbacks(self):
        self.update_callback('capture_mode_scanning', lambda v: self._set_camera_mode(v))

        mode = image_capture.texture_mode
        self.update_callback('brightness_texture_scanning', mode.set_brightness)
        self.update_callback('contrast_texture_scanning', mode.set_contrast)
        self.update_callback('saturation_texture_scanning', mode.set_saturation)
        self.update_callback('exposure_texture_scanning', mode.set_exposure)
        self.update_callback('light1_texture_scanning', lambda v: mode.set_light(1,v) )
        self.update_callback('light2_texture_scanning', lambda v: mode.set_light(2,v) )

        mode = image_capture.laser_mode
        self.update_callback('brightness_laser_scanning', mode.set_brightness)
        self.update_callback('contrast_laser_scanning', mode.set_contrast)
        self.update_callback('saturation_laser_scanning', mode.set_saturation)
        self.update_callback('exposure_laser_scanning', mode.set_exposure)
        self.update_callback('light1_laser_scanning', lambda v: mode.set_light(1,v) )
        self.update_callback('light2_laser_scanning', lambda v: mode.set_light(2,v) )
        self.update_callback('remove_background_scanning', image_capture.set_remove_background)

    def on_selected(self):
        current_video.updating = True
        current_video.sync()

        # Update mode settings
        current_video.calibration = False
        current_video.mode = profile.settings['capture_mode_scanning']

        image_capture.texture_mode.read_profile('texture_scanning')
        image_capture.laser_mode.read_profile('laser_scanning')
        image_capture.set_remove_background(profile.settings['remove_background_scanning'])

        profile.settings['current_video_mode_adjustment'] = current_video.mode
        profile.settings['current_panel_adjustment'] = 'scan_capture'

        current_video.flush()
        current_video.updating = False

    def _set_camera_mode(self, mode):
        current_video.updating = True
        current_video.sync()
        # Update mode settings
        self._set_mode_layout(mode)
        current_video.mode = mode
        profile.settings['current_video_mode_adjustment'] = current_video.mode
        current_video.flush()
        current_video.updating = False

    def _set_mode_layout(self, mode):
        if mode == 'Laser':
            self.get_control('brightness_texture_scanning').Hide()
            self.get_control('contrast_texture_scanning').Hide()
            self.get_control('saturation_texture_scanning').Hide()
            self.get_control('exposure_texture_scanning').Hide()
            self.get_control('light1_texture_scanning').Hide()
            self.get_control('light2_texture_scanning').Hide()

            self.get_control('brightness_laser_scanning').Show()
            self.get_control('contrast_laser_scanning').Show()
            self.get_control('saturation_laser_scanning').Show()
            self.get_control('exposure_laser_scanning').Show()
            self.get_control('light1_laser_scanning').Show()
            self.get_control('light2_laser_scanning').Show()
            self.get_control('remove_background_scanning').Show()

        elif mode == 'Texture':
            self.get_control('brightness_texture_scanning').Show()
            self.get_control('contrast_texture_scanning').Show()
            self.get_control('saturation_texture_scanning').Show()
            self.get_control('exposure_texture_scanning').Show()
            self.get_control('light1_texture_scanning').Show()
            self.get_control('light2_texture_scanning').Show()

            self.get_control('brightness_laser_scanning').Hide()
            self.get_control('contrast_laser_scanning').Hide()
            self.get_control('saturation_laser_scanning').Hide()
            self.get_control('exposure_laser_scanning').Hide()
            self.get_control('light1_laser_scanning').Hide()
            self.get_control('light2_laser_scanning').Hide()
            self.get_control('remove_background_scanning').Hide()

        if sys.is_wx30():
            self.content.SetSizerAndFit(self.content.vbox)
        if sys.is_windows():
            self.parent.Refresh()
        self.parent.Layout()
        self.Layout()


    def laser_reset(self):
        current_video.updating = True
        current_video.sync()

        profile.laser_bg_scanning = [None for _ in profile.laser_bg_scanning]
        image_capture.laser_mode.laser_bg = profile.laser_bg_scanning

        current_video.updating = False


    def laser_capture(self):
        if current_video.updating:
            return
        current_video.updating = True
        current_video.sync()

        dlg = wx.MessageDialog(
            self,
            _("Please remove everything from platform and press OK."),
            _("Capture laser background"), wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

        image_capture.laser_mode.laser_bg_enable = False
        images = image_capture.capture_lasers()[:-1]
        for i, image in enumerate(images):
            images[i] = laser_segmentation.compute_line_segmentation_bg(image, self.avoid_platform)

        profile.laser_bg_scanning = images
        image_capture.laser_mode.laser_bg = images
        image_capture.laser_mode.laser_bg_enable = profile.laser_bg_scanning_enable

        current_video.updating = False


    def laser_bg_enable(self, value):
        current_video.updating = True
        current_video.sync()

        profile.laser_bg_scanning_enable = value
        image_capture.laser_mode.laser_bg_enable = profile.laser_bg_scanning_enable

        current_video.updating = False


    def set_avoid_platform(self, value):
        self.avoid_platform = value


class ScanSegmentationPanel(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(self, parent, _("Scan segmentation"))

    def add_controls(self):
        # self.add_control('laser_color_detector_scanning', ComboBox)
        self.add_control('draw_line_scanning', CheckBox)
        self.add_control('laser_color_detector_scanning', ComboBox, _("Laser color detection algorithm"))
        self.add_control(
            'threshold_value_scanning', Slider,
            _("Remove all pixels which intensity is less that the threshold value"))
        self.add_control(
            'threshold_enable_scanning', CheckBox,
            _("Remove all pixels which intensity is less that the threshold value"))
        self.add_control(
            'blur_value_scanning', Slider,
            _("Blur with Normalized box filter. Kernel size: 2 * value + 1"))
        self.add_control(
            'blur_enable_scanning', CheckBox,
            _("Blur with Normalized box filter. Kernel size: 2 * value + 1"))
        self.add_control(
            'window_value_scanning', Slider,
            _("Filter pixels out of 2 * window value around the intensity peak"))
        self.add_control(
            'window_enable_scanning', CheckBox,
            _("Filter pixels out of 2 * window value around the intensity peak"))
        self.add_control('refinement_scanning', ComboBox)

    def update_callbacks(self):
        # self.update_callback('laser_color_detector_scanning', laser_segmentation.set_laser_color_detector)
        self.update_callback('draw_line_scanning', current_video.set_draw_line)
        self.update_callback('laser_color_detector_scanning', laser_segmentation.set_laser_color_detector)
        self.update_callback('threshold_value_scanning', laser_segmentation.set_threshold_value)
        self.update_callback('threshold_enable_scanning', laser_segmentation.set_threshold_enable)
        self.update_callback('blur_value_scanning', laser_segmentation.set_blur_value)
        self.update_callback('blur_enable_scanning', laser_segmentation.set_blur_enable)
        self.update_callback('window_value_scanning', laser_segmentation.set_window_value)
        self.update_callback('window_enable_scanning', laser_segmentation.set_window_enable)
        self.update_callback('refinement_scanning', laser_segmentation.set_refinement_method)

    def on_selected(self):
        current_video.updating = True
        current_video.sync()

        # Update mode settings
        current_video.calibration = False
        current_video.mode = 'Gray'
        current_video.set_draw_line(profile.settings['draw_line_scanning'])

        image_capture.laser_mode.read_profile('laser_scanning')
        image_capture.set_remove_background(profile.settings['remove_background_scanning'])

        laser_segmentation.read_profile('scanning')

        profile.settings['current_video_mode_adjustment'] = current_video.mode
        profile.settings['current_panel_adjustment'] = 'scan_segmentation'
        current_video.flush()
        current_video.updating = False


class CalibrationCapturePanel(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        self.avoid_platform = False
        ExpandablePanel.__init__(self, parent, _("Calibration capture"))

    def add_controls(self):
        self.add_control('capture_mode_calibration', ComboBox)
        self.add_control(
            'brightness_pattern_calibration', Slider,
            _("Image luminosity. Low values are better for environments with high ambient "
              "light conditions. High values are recommended for poorly lit places"))
        self.add_control(
            'contrast_pattern_calibration', Slider,
            _("Relative difference in intensity between an image point and its "
              "surroundings. Low values are recommended for black or very dark colored "
              "objects. High values are better for very light colored objects"))
        self.add_control(
            'saturation_pattern_calibration', Slider,
            _("Purity of color. Low values will cause colors to disappear from the image. "
              "High values will show an image with very intense colors"))
        self.add_control(
            'exposure_pattern_calibration', Slider,
            _("Amount of light per unit area. It is controlled by the time the camera "
              "sensor is exposed during a frame capture. "
              "High values are recommended for poorly lit places"))

        self.add_control(
            'light1_pattern_calibration', Slider,
            _("Photo lamp 1 brightness"))
        self.add_control(
            'light2_pattern_calibration', Slider,
            _("Photo lamp 2 brightness"))


        self.add_control(
            'brightness_laser_calibration', Slider,
            _("Image luminosity. Low values are better for environments with high ambient "
              "light conditions. High values are recommended for poorly lit places"))
        self.add_control(
            'contrast_laser_calibration', Slider,
            _("Relative difference in intensity between an image point and its "
              "surroundings. Low values are recommended for black or very dark colored "
              "objects. High values are better for very light colored objects"))
        self.add_control(
            'saturation_laser_calibration', Slider,
            _("Purity of color. Low values will cause colors to disappear from the image. "
              "High values will show an image with very intense colors"))
        self.add_control(
            'exposure_laser_calibration', Slider,
            _("Amount of light per unit area. It is controlled by the time the camera "
              "sensor is exposed during a frame capture. "
              "High values are recommended for poorly lit places"))

        self.add_control(
            'light1_laser_calibration', Slider,
            _("Photo lamp 1 brightness"))
        self.add_control(
            'light2_laser_calibration', Slider,
            _("Photo lamp 2 brightness"))

        self.add_control(
            'remove_background_calibration', CheckBox,
            _("Capture an extra image without laser to remove "
              "the background in the laser's image"))

        # Initial layout
        self._set_mode_layout(profile.settings['capture_mode_calibration'])

        # capture bg buttons panel
        control = wx.Panel(self.content)
        control.SetToolTip(wx.ToolTip("Laser background filter"))

        label = wx.StaticText(control, label=_("Laser background filter"), style=wx.ALIGN_CENTER)
        enable_box = wx.CheckBox(control, label="Enable filter")
        platform_box = wx.CheckBox(control, label="Don't mask platform")
        buttons_box = wx.Panel(control)
        buttons_box.left_button = wx.Button(buttons_box, -1, "Reset")
        buttons_box.right_button = wx.Button(buttons_box, -1, "Capture")

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(label, 0, wx.TOP | wx.EXPAND, border=5)
        vbox.Add(enable_box, 0, wx.TOP | wx.EXPAND, border=5)
        vbox.Add(platform_box, 0, wx.TOP | wx.EXPAND, border=5)
        vbox.Add(buttons_box, 0, wx.TOP | wx.EXPAND, border=5)
        control.SetSizer(vbox)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(buttons_box.left_button, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        hbox.AddStretchSpacer()
        hbox.Add(buttons_box.right_button, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        buttons_box.SetSizer(hbox)

        control.Layout()

        # Events
        buttons_box.left_button.Bind(wx.EVT_BUTTON, lambda v: self.laser_reset() )
        buttons_box.right_button.Bind(wx.EVT_BUTTON, lambda v: self.laser_capture() )
        enable_box.Bind(wx.EVT_CHECKBOX, lambda v: self.laser_bg_enable(v.GetEventObject().GetValue()) )
        platform_box.Bind(wx.EVT_CHECKBOX, lambda v: self.set_avoid_platform(v.GetEventObject().GetValue()) )

        enable_box.SetValue(profile.laser_bg_calibration_enable)
        platform_box.SetValue(self.avoid_platform)

        self.content.vbox.Add(control, 0, wx.BOTTOM | wx.EXPAND, 5)
        self.content.vbox.Layout()
        if sys.is_wx30():
            self.content.SetSizerAndFit(self.content.vbox)


    def update_callbacks(self):
        self.update_callback('capture_mode_calibration', lambda v: self._set_camera_mode(v))

        mode = image_capture.pattern_mode
        self.update_callback('brightness_pattern_calibration', mode.set_brightness)
        self.update_callback('contrast_pattern_calibration', mode.set_contrast)
        self.update_callback('saturation_pattern_calibration', mode.set_saturation)
        self.update_callback('exposure_pattern_calibration', mode.set_exposure)
        self.update_callback('light1_pattern_calibration', lambda v: mode.set_light(1,v) )
        self.update_callback('light2_pattern_calibration', lambda v: mode.set_light(2,v) )

        mode = image_capture.laser_mode
        self.update_callback('brightness_laser_calibration', mode.set_brightness)
        self.update_callback('contrast_laser_calibration', mode.set_contrast)
        self.update_callback('saturation_laser_calibration', mode.set_saturation)
        self.update_callback('exposure_laser_calibration', mode.set_exposure)
        self.update_callback('light1_laser_calibration', lambda v: mode.set_light(1,v) )
        self.update_callback('light2_laser_calibration', lambda v: mode.set_light(2,v) )
        self.update_callback('remove_background_calibration', image_capture.set_remove_background)

    def on_selected(self):
        current_video.updating = True
        current_video.sync()

        # Update mode settings
        current_video.calibration = True
        current_video.mode = profile.settings['capture_mode_calibration']

        image_capture.pattern_mode.read_profile('pattern_calibration')
        image_capture.laser_mode.read_profile('laser_calibration')
        image_capture.set_remove_background(profile.settings['remove_background_calibration'])

        profile.settings['current_video_mode_adjustment'] = current_video.mode
        profile.settings['current_panel_adjustment'] = 'calibration_capture'
        current_video.flush()
        current_video.updating = False

    def _set_camera_mode(self, mode):
        current_video.updating = True
        current_video.sync()
        # Update mode settings
        self._set_mode_layout(mode)
        current_video.mode = mode
        profile.settings['current_video_mode_adjustment'] = current_video.mode
        current_video.flush()
        current_video.updating = False

    def _set_mode_layout(self, mode):
        if mode == 'Laser':
            self.get_control('brightness_pattern_calibration').Hide()
            self.get_control('contrast_pattern_calibration').Hide()
            self.get_control('saturation_pattern_calibration').Hide()
            self.get_control('exposure_pattern_calibration').Hide()
            self.get_control('light1_pattern_calibration').Hide()
            self.get_control('light2_pattern_calibration').Hide()

            self.get_control('brightness_laser_calibration').Show()
            self.get_control('contrast_laser_calibration').Show()
            self.get_control('saturation_laser_calibration').Show()
            self.get_control('exposure_laser_calibration').Show()
            self.get_control('light1_laser_calibration').Show()
            self.get_control('light2_laser_calibration').Show()
            self.get_control('remove_background_calibration').Show()

        elif mode == 'Pattern':
            self.get_control('brightness_pattern_calibration').Show()
            self.get_control('contrast_pattern_calibration').Show()
            self.get_control('saturation_pattern_calibration').Show()
            self.get_control('exposure_pattern_calibration').Show()
            self.get_control('light1_pattern_calibration').Show()
            self.get_control('light2_pattern_calibration').Show()

            self.get_control('brightness_laser_calibration').Hide()
            self.get_control('contrast_laser_calibration').Hide()
            self.get_control('saturation_laser_calibration').Hide()
            self.get_control('exposure_laser_calibration').Hide()
            self.get_control('light1_laser_calibration').Hide()
            self.get_control('light2_laser_calibration').Hide()
            self.get_control('remove_background_calibration').Hide()

        if sys.is_wx30():
            self.content.SetSizerAndFit(self.content.vbox)
        if sys.is_windows():
            self.parent.Refresh()
        self.parent.Layout()
        self.Layout()


    def laser_reset(self):
        current_video.updating = True
        current_video.sync()

        profile.laser_bg_calibration = [None for _ in profile.laser_bg_calibration]
        image_capture.laser_mode.laser_bg = profile.laser_bg_calibration

        current_video.updating = False


    def laser_capture(self):
        if current_video.updating:
            return
        current_video.updating = True
        current_video.sync()

        dlg = wx.MessageDialog(
            self,
            _("Please remove everything from platform and press OK."),
            _("Capture laser background"), wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

        image_capture.laser_mode.laser_bg_enable = False
        images = image_capture.capture_lasers()[:-1]
        for i, image in enumerate(images):
            images[i] = laser_segmentation.compute_line_segmentation_bg(image, self.avoid_platform)

        profile.laser_bg_calibration = images
        image_capture.laser_mode.laser_bg = images
        image_capture.laser_mode.laser_bg_enable = profile.laser_bg_calibration_enable

        current_video.updating = False


    def laser_bg_enable(self, value):
        current_video.updating = True
        current_video.sync()

        profile.laser_bg_calibration_enable = value
        image_capture.laser_mode.laser_bg_enable = profile.laser_bg_calibration_enable

        current_video.updating = False


    def set_avoid_platform(self, value):
        self.avoid_platform = value


class CalibrationSegmentationPanel(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(self, parent, _("Calibration segmentation"))

    def add_controls(self):
        # self.add_control('laser_color_detector_calibration', ComboBox)
        self.add_control('draw_line_calibration', CheckBox)
        self.add_control('laser_color_detector_calibration', ComboBox, _("Laser color detection algorithm"))
        self.add_control(
            'threshold_value_calibration', Slider,
            _("Remove all pixels which intensity is less that the threshold value"))
        self.add_control(
            'threshold_enable_calibration', CheckBox,
            _("Remove all pixels which intensity is less that the threshold value"))
        self.add_control(
            'blur_value_calibration', Slider,
            _("Blur with Normalized box filter. Kernel size: 2 * value + 1"))
        self.add_control(
            'blur_enable_calibration', CheckBox,
            _("Blur with Normalized box filter. Kernel size: 2 * value + 1"))
        self.add_control(
            'window_value_calibration', Slider,
            _("Filter pixels out of 2 * window value around the intensity peak"))
        self.add_control(
            'window_enable_calibration', CheckBox,
            _("Filter pixels out of 2 * window value around the intensity peak"))
        self.add_control('refinement_calibration', ComboBox)

    def update_callbacks(self):
        # self.update_callback('laser_color_detector_calibration', laser_segmentation.set_laser_color_detector)
        self.update_callback('draw_line_calibration', current_video.set_draw_line)
        self.update_callback('laser_color_detector_calibration', laser_segmentation.set_laser_color_detector)
        self.update_callback('threshold_value_calibration', laser_segmentation.set_threshold_value)
        self.update_callback(
            'threshold_enable_calibration', laser_segmentation.set_threshold_enable)
        self.update_callback('blur_value_calibration', laser_segmentation.set_blur_value)
        self.update_callback('blur_enable_calibration', laser_segmentation.set_blur_enable)
        self.update_callback('window_value_calibration', laser_segmentation.set_window_value)
        self.update_callback('window_enable_calibration', laser_segmentation.set_window_enable)
        self.update_callback('refinement_calibration', laser_segmentation.set_refinement_method)

    def on_selected(self):
        current_video.updating = True
        current_video.sync()

        # Update mode settings
        current_video.calibration = True
        current_video.mode = 'Gray'
        current_video.set_draw_line(profile.settings['draw_line_calibration'])

        image_capture.laser_mode.read_profile('laser_calibration')
        image_capture.set_remove_background(profile.settings['remove_background_calibration'])

        laser_segmentation.read_profile('calibration')

        profile.settings['current_video_mode_adjustment'] = current_video.mode
        profile.settings['current_panel_adjustment'] = 'calibration_segmentation'
        current_video.flush()
        current_video.updating = False
