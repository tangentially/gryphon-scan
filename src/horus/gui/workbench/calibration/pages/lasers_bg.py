# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import random
import wx._core
import numpy as np

from horus.util import profile

from horus.gui.workbench.calibration.pages.page import Page
from horus.gui.workbench.calibration.pages.lasers_bg_capture import LasersBgCapturePage


class LasersBgPages(wx.Panel):

    def __init__(self, parent, start_callback=None, exit_callback=None):
        wx.Panel.__init__(self, parent)

        # Elements
        self.capture_page = LasersBgCapturePage(self)

        # Layout
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.capture_page, 1, wx.ALL | wx.EXPAND, 0)
        self.SetSizer(hbox)
        self.Layout()

        self._initialize()

    def _initialize(self):
        self.capture_page.SetFocus()
        self.capture_page.Show()
        self.capture_page.left_button.Enable()

    def play(self):
        self.capture_page.play()

    def stop(self):
        self.capture_page.stop()

    def reset(self):
        self.capture_page.reset()

