# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core
import cv2

from horus.util import resources

from horus.util import profile
from horus.gui.engine import image_capture, image_detection, laser_segmentation
from horus.gui.workbench.calibration.pages.page import Page
from horus.gui.util.image_view import ImageView
from horus.gui.util.video_view import VideoView


class LasersBgCapturePage(Page):

    def __init__(self, parent, start_callback=None):
        Page.__init__(self, parent,
                      title=_("Laser segmentation of environment, lasers background capture (advanced)"),
                      desc=_("Pleas remove everything from platform and press \'Start\'"),
                      left=_("Reset"),
                      right=_("Start"),
                      button_left_callback=self.reset_bg,
                      button_right_callback=self.capture_bg,
                      view_progress=True)

#        self.right_button.Hide()
        self.gauge.Hide()

        # Elements
        self.video_view = VideoView(self.panel, self.get_image)
        self.image_grid_panel = wx.Panel(self.panel)
        self.grid_sizer = wx.GridSizer(1, len(image_capture.laser_bg), 3, 3)

        self.panel_grid = []
        for bg in image_capture.laser_bg:
            i = ImageView(self.image_grid_panel)
            i.SetBackgroundColour((221, 221, 221))
            if bg is not None:
                i.set_frame(bg)
            else:
                i.set_image(wx.Image(resources.get_path_for_image("void.png")))
            self.grid_sizer.Add(i, 0, wx.ALL | wx.EXPAND)
            self.panel_grid.append(i)
        self.image_grid_panel.SetSizer(self.grid_sizer)

        # Layout
        self.panel_box.Add(self.video_view, 1, wx.ALL | wx.EXPAND, 2)
        self.panel_box.Add(self.image_grid_panel, len(self.panel_grid), wx.ALL | wx.EXPAND, 3)
        self.Layout()

    def play(self):
        self.video_view.play()
        self.image_grid_panel.SetFocus()
        self.GetParent().Layout()
        self.Layout()

    def stop(self):
        self.video_view.stop()

    def reset(self):
        self.video_view.reset()

    def get_image(self):
        image = image_capture.capture_all_lasers()
        return image

    def reset_bg(self):
        image_capture.laser_bg = [None for _ in image_capture.laser_bg]
        for panel in self.panel_grid:
            panel.SetBackgroundColour((221, 221, 221))
            panel.set_image(wx.Image(resources.get_path_for_image("void.png")))

    def capture_bg(self):
            self.video_view.stop()
            self.left_button.Disable()
            self.right_button.Disable()

            self.reset_bg()
            images = image_capture.capture_lasers()
            for i, image in enumerate(images):
                image = laser_segmentation.compute_line_segmentation_bg(image)
                self.panel_grid[i].set_frame(image)

            image_capture.laser_bg = images
            profile.laser_bg_scan = images

            self.left_button.Enable()
            self.right_button.Enable()
            self.video_view.play()

