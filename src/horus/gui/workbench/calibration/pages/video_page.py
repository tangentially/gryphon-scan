# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core

from horus.util import resources

from horus.gui.engine import image_capture, image_detection, scanner_autocheck, laser_triangulation, \
    platform_extrinsics
from horus.gui.workbench.calibration.pages.page import Page
from horus.gui.util.image_view import ImageView
from horus.gui.util.video_view import VideoView
from horus.gui.util.augmented_view import augmented_draw_platform, augmented_draw_lasers_on_platform, \
    augmented_draw_lasers_on_pattern


class VideoPage(Page):

    def __init__(self, parent, title='Video page', start_callback=None, cancel_callback=None):
        Page.__init__(self, parent,
                      title="",
                      desc="",
                      left=_("Cancel"),
                      right=_("Start"),
                      button_left_callback=cancel_callback,
                      button_right_callback=start_callback,
                      view_progress=True)

        # Elements
        self.video_view = VideoView(self.panel, self.get_image)

        self.info_panel = wx.Panel(self)
        title_text = wx.StaticText(self.info_panel, label=title)
        title_font = title_text.GetFont()
        title_font.SetWeight(wx.BOLD)
        title_text.SetFont(title_font)

        # Layout
        self.info_box = wx.BoxSizer(wx.VERTICAL)
        self.info_panel.SetSizer(self.info_box)
        if title_text != "":
            self.info_box.Add(title_text, 0, wx.ALL ^ wx.BOTTOM | wx.EXPAND, 12)

        self.panel_box.Add(self.info_panel, 2, wx.ALL | wx.EXPAND, 3)
        self.panel_box.Add(self.video_view, 2, wx.ALL | wx.EXPAND, 3)

        self.Layout()

#	self.add_info(_("Put the pattern on the platform as shown in the "
#                             "picture and press \"Start\""), "pattern-position.png")

    def add_info(self, desc, picture):
        if desc != "":
            desc_text = wx.StaticText(self.info_panel, label=desc)
            self.info_box.Add(desc_text, 0, wx.ALL | wx.EXPAND, 14)

        if picture != "":
            image_view = ImageView(self.info_panel, quality=wx.IMAGE_QUALITY_HIGH)
            image_view.set_image(wx.Image(resources.get_path_for_image(picture)))
            self.info_box.Add(image_view, 1, wx.ALL | wx.EXPAND, 3)

        self.Layout()

    def initialize(self):
        self.gauge.SetValue(0)
        self.right_button.Enable()

    def play(self):
        self.video_view.play()
        self.GetParent().Layout()
        self.Layout()

    def stop(self):
        self.initialize()
        self.video_view.stop()

    def reset(self):
        self.video_view.reset()

    def get_image(self):
        if scanner_autocheck.image is not None:
            image = scanner_autocheck.image
        elif laser_triangulation.image is not None:
            image = laser_triangulation.image
        elif platform_extrinsics.image is not None:
            image = platform_extrinsics.image
        else:
            image = image_capture.capture_pattern()
            if image is not None:
                corners = image_detection.detect_corners(image)
                image = image_detection.draw_pattern(image, corners)
                pose = image_detection.detect_pose_from_corners(corners)
                augmented_draw_lasers_on_pattern(image, pose)
            augmented_draw_lasers_on_platform(image)

        augmented_draw_platform(image)
        return image
