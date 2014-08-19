#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014 Mundo Reader S.L.                                  #
#                                                                       #
# Date: June 2014                                                       #
# Author: Jesús Arroyo Torrens <jesus.arroyo@bq.com>                    #
#                                                                       #
# This program is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published by  #
# the Free Software Foundation, either version 3 of the License, or     #
# (at your option) any later version.                                   #
#                                                                       #
# This program is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of        #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
# GNU General Public License for more details.                          #
#                                                                       #
# You should have received a copy of the GNU General Public License     #
# along with this program. If not, see <http://www.gnu.org/licenses/>.  #
#                                                                       #
#-----------------------------------------------------------------------#

__author__ = "Jesús Arroyo Torrens <jesus.arroyo@bq.com>"
__license__ = "GNU General Public License v3 http://www.gnu.org/licenses/gpl.html"

from horus.util.resources import *

from horus.gui.util.cameraPanel import *
from horus.gui.util.videoView import *
from horus.gui.util.devicePanel import *
from horus.gui.util.videoView import *
from horus.gui.util.workbenchConnection import *

class ControlWorkbench(WorkbenchConnection):

	def __init__(self, parent):
		WorkbenchConnection.__init__(self, parent, 0, 1)

		self.viewCamera = True

		self.load()

		self.laserLeft = False
		self.laserRight = False

		self.timer = wx.Timer(self)
		self.Bind(wx.EVT_TIMER, self.onTimer, self.timer)

		self.Bind(wx.EVT_SHOW, self.onShow)

	def load(self):
		#-- Toolbar Configuration
		self.playTool          = self.toolbar.AddLabelTool(wx.NewId(), _("Play"), wx.Bitmap(getPathForImage("play.png")), shortHelp=_("Play"))
		self.stopTool          = self.toolbar.AddLabelTool(wx.NewId(), _("Stop"), wx.Bitmap(getPathForImage("stop.png")), shortHelp=_("Stop"))
		self.snapshotTool      = self.toolbar.AddLabelTool(wx.NewId(), _("Snapshot"), wx.Bitmap(getPathForImage("snapshot.png")), shortHelp=_("Snapshot"))
		self.viewTool          = self.toolbar.AddLabelTool(wx.NewId(), _("View"), wx.Bitmap(getPathForImage("view.png")), shortHelp=_("Camera / Device"))
		self.toolbar.Realize()

		#-- Disable Toolbar Items
		self.enableLabelTool(self.playTool    , False)
		self.enableLabelTool(self.stopTool    , False)
		self.enableLabelTool(self.snapshotTool, False)
		self.enableLabelTool(self.viewTool    , True)

		#-- Bind Toolbar Items
		self.Bind(wx.EVT_TOOL, self.onPlayToolClicked    , self.playTool)
		self.Bind(wx.EVT_TOOL, self.onStopToolClicked    , self.stopTool)
		self.Bind(wx.EVT_TOOL, self.onSnapshotToolClicked, self.snapshotTool)
		self.Bind(wx.EVT_TOOL, self.onViewToolClicked    , self.viewTool)

		#-- Left Panel
		self.cameraPanel = CameraPanel(self._leftPanel)
		self.devicePanel = DevicePanel(self._leftPanel)

		#-- Right Views
		self.cameraView = VideoView(self._rightPanel)
		self.deviceView = VideoView(self._rightPanel)
		self.cameraView.SetBackgroundColour(wx.BLACK)

		self.addToLeft(self.cameraPanel)
		self.addToRight(self.cameraView)

		self.addToLeft(self.devicePanel)
		self.addToRight(self.deviceView)

		self.deviceView.setImage(wx.Image(getPathForImage("scanner.png")))

		self.updateView()

	def onShow(self, event):
		if not event.GetShow():
			self.onStopToolClicked(None)

	def onTimer(self, event):
		frame = self.scanner.camera.captureImage()
		if frame is not None:
			self.cameraView.setFrame(frame)

	def onPlayToolClicked(self, event):
		self.enableLabelTool(self.playTool, False)
		self.enableLabelTool(self.stopTool, True)
		mseconds= 1000/self.scanner.camera.fps
		self.timer.Start(milliseconds=mseconds)

	def onStopToolClicked(self, event):
		self.enableLabelTool(self.playTool, True)
		self.enableLabelTool(self.stopTool, False)
		self.timer.Stop()
		self.cameraView.setDefaultImage()

	def onSnapshotToolClicked(self, event):
		frame = self.scanner.camera.captureImage()
		if frame is not None:
			self.cameraView.setFrame(frame)

	def onViewToolClicked(self, event):
		self.viewCamera = not self.viewCamera
		profile.putPreference('view_camera', self.viewCamera)
		self.updateView()

	def updateView(self):
		if self.viewCamera:
			self.cameraPanel.Show()
			self.cameraView.Show()
			self.devicePanel.Hide()
			self.deviceView.Hide()
		else:
			self.cameraPanel.Hide()
			self.cameraView.Hide()
			self.devicePanel.Show()
			self.deviceView.Show()
		self.Layout()

	def updateToolbarStatus(self, status):
		if status:
			self.enableLabelTool(self.playTool    , True)
			self.enableLabelTool(self.stopTool    , False)
			self.enableLabelTool(self.snapshotTool, True)
			self.laserLeft = False
			self.laserRight = False
		else:
			self.enableLabelTool(self.playTool    , False)
			self.enableLabelTool(self.stopTool    , False)
			self.enableLabelTool(self.snapshotTool, False)

	def updateProfileToAllControls(self):
		self.cameraPanel.updateProfileToAllControls()
		self.devicePanel.updateProfileToAllControls()
		self.viewCamera = profile.getPreferenceBool('view_camera')
		self.updateView()