# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core

from horus.util import resources

from horus.gui.engine import image_capture, image_detection
from horus.gui.util.image_view import ImageView
from horus.gui.util.video_view import VideoView
from horus.gui.util.augmented_view import augmented_draw_pattern

class PatternSettingsPages(wx.Panel):

    def __init__(self, parent, start_callback=None, exit_callback=None):
        wx.Panel.__init__(self, parent)

        # Elements
        self.video_view = VideoView(self, self.get_image)

        self.info_panel = wx.Panel(self)
        title_text = wx.StaticText(self.info_panel, label="Pattern settings")
        title_font = title_text.GetFont()
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        title_text.SetFont(title_font)

        # Layout
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(hbox)

        self.info_box = wx.BoxSizer(wx.VERTICAL)
        self.info_panel.SetSizer(self.info_box)
        self.info_box.Add(title_text, 0, wx.ALL ^ wx.BOTTOM | wx.EXPAND, 12)

        hbox.Add(self.info_panel, 1, wx.ALL | wx.EXPAND, 3)
        hbox.Add(self.video_view, 1, wx.ALL | wx.EXPAND, 3)

        self.add_info(_("1) Pattern size is the number of inner \"cross\" points "
                                 "of the pattern"), "pattern-size.jpg")

        self.add_info(_("2) Origin distance is the distance from lower set of  \"cross\" points "
                                 "to the platform"), "pattern-distance.jpg")

        self.add_info(_("3) Pattern border is the clean white space around chessboard pattern."
                                 "\nThis white space is used for lasers calibration"), "")

        self.Layout()

    def add_info(self, desc, picture):
        if desc != "":
            desc_text = wx.StaticText(self.info_panel, label=desc)
            self.info_box.Add(desc_text, 0, wx.ALL | wx.EXPAND, 14)

        if picture != "":
            image_view = ImageView(self.info_panel, quality=wx.IMAGE_QUALITY_HIGH)
            image_view.set_image(wx.Image(resources.get_path_for_image(picture)))
            self.info_box.Add(image_view, 1, wx.ALL | wx.EXPAND, 3)


    def play(self):
        self.video_view.play()
        self.GetParent().Layout()
        self.Layout()

    def stop(self):
        self.video_view.stop()

    def reset(self):
        self.video_view.reset()

    @staticmethod
    def get_image():
        image_capture.stream = True
        image = image_capture.capture_pattern()
        corners = image_detection.detect_corners(image)

        augmented_draw_pattern(image, corners)

        return image
