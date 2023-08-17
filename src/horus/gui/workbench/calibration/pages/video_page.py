# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core

from horus.util import resources
import numpy as np
import cv2

from horus.gui.engine import image_capture, image_detection, scanner_autocheck, \
    laser_triangulation, platform_extrinsics, calibration_data

from horus.engine.algorithms.aruco_detection import aruco_detection

from horus.gui.workbench.calibration.pages.page import Page
from horus.gui.util.image_view import ImageView
from horus.gui.util.video_view import VideoView
from horus.gui.util.augmented_view import augmented_draw_platform, augmented_draw_lasers_on_platform, \
    augmented_draw_lasers_on_pattern
from horus.util.gryphon_util  import rotatePoint2Plane


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
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        title_text.SetFont(title_font)

        # Layout
        self.info_box = wx.BoxSizer(wx.VERTICAL)
        self.info_panel.SetSizer(self.info_box)
        if title_text != "":
            self.info_box.Add(title_text, 0, wx.ALL ^ wx.BOTTOM | wx.EXPAND, 12)

        self.panel_box.Add(self.info_panel, 2, wx.ALL | wx.EXPAND, 3)
        self.panel_box.Add(self.video_view, 4, wx.ALL | wx.EXPAND, 3)

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

    @staticmethod
    def get_image():
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

                if pose is not None:
                    for i,laser in enumerate(calibration_data.laser_planes):
                        if not laser.is_empty():
                            #print(i)
                            l = rotatePoint2Plane(pose[1].T[0], laser.normal, laser.distance)
                            #print(l)
                            if l is not None:
                                cv2.putText(image, str(i)+": "+str(np.round(l,2)), (10,80+i*30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), lineType=cv2.LINE_AA)
                    P = calibration_data.platform_translation # platform center
                    M = calibration_data.platform_rotation    # platform to world matrix
                    A = M.T.dot(pose[1].T[0] - P) # A in platform coords
                    cv2.putText(image, "R: "+str(np.round(np.linalg.norm(A[0:2]),2)), (10,160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), lineType=cv2.LINE_AA)

                corners, ids = aruco_detection.aruco_detect(image)
                if corners:
                    image,_,_ = aruco_detection.aruco_draw_markers(image, corners, ids)

        augmented_draw_platform(image)
        return image
