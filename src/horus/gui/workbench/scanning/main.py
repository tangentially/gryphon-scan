# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core

from horus.util import resources, profile

from horus.engine.driver.camera import InputOutputError

from horus.gui.engine import driver, image_capture, laser_segmentation, calibration_data, \
    ciclop_scan, current_video, point_cloud_roi
from horus.gui.workbench.workbench import Workbench
from horus.gui.workbench.scanning.view_page import ViewPage
from horus.gui.workbench.scanning.panels import ScanParameters, RotatingPlatform, \
    PointCloudROI
from horus.gui.workbench.scanning.gryphon_panels import PointCloudColor, Photogrammetry, \
    MeshCorrection


class ScanningWorkbench(Workbench):

    def __init__(self, parent, toolbar_scan):
        Workbench.__init__(self, parent, name=_('Scanning workbench'))

        self.scanning = False
        self.toolbar_scan = toolbar_scan

        # Elements
        self.play_tool = self.toolbar_scan.AddLabelTool(
            wx.NewId(), _("Play"),
            wx.Bitmap(resources.get_path_for_image("play.png")), shortHelp=_("Play"))
        self.stop_tool = self.toolbar_scan.AddLabelTool(
            wx.NewId(), _("Stop"),
            wx.Bitmap(resources.get_path_for_image("stop.png")), shortHelp=_("Stop"))
        self.pause_tool = self.toolbar_scan.AddLabelTool(
            wx.NewId(), _("Pause"),
            wx.Bitmap(resources.get_path_for_image("pause.png")), shortHelp=_("Pause"))
        self.toolbar_scan.Realize()
        self.toolbar_scan.GetParent().Layout()

        ciclop_scan.point_cloud_callback = self.point_cloud_callback

        self._enable_tool_scan(self.play_tool, False)
        self._enable_tool_scan(self.stop_tool, False)
        self._enable_tool_scan(self.pause_tool, False)

        # Events
        self.toolbar_scan.GetParent().Bind(wx.EVT_TOOL, self.on_play_tool_clicked, self.play_tool)
        self.toolbar_scan.GetParent().Bind(wx.EVT_TOOL, self.on_stop_tool_clicked, self.stop_tool)
        self.toolbar_scan.GetParent().Bind(wx.EVT_TOOL, self.on_pause_tool_clicked, self.pause_tool)

    def add_panels(self):
        self.add_panel('scan_parameters', ScanParameters)
        self.add_panel('rotating_platform', RotatingPlatform)
        self.add_panel('point_cloud_roi', PointCloudROI)
        self.add_panel('point_cloud_color', PointCloudColor)
        self.add_panel('photogrammetry', Photogrammetry)
        self.add_panel('mesh_correction',MeshCorrection)

    def add_pages(self):
        self.add_page('view_page', ViewPage(self, self.get_image))
        self.video_view = self.pages_collection['view_page'].video_view
        self.scene_panel = self.pages_collection['view_page'].scene_panel
        self.scene_view = self.pages_collection['view_page'].scene_view
        self.gauge = self.pages_collection['view_page'].gauge
        self.panels_collection.expandable_panels[
            profile.settings['current_panel_scanning']].on_title_clicked(None)

    def on_open(self):
        if self.video_view.IsShown():
            self.video_view.play()
        if driver.is_connected and profile.settings['current_panel_scanning'] == 'point_cloud_roi':
            self.scene_view._view_roi = profile.settings['use_roi']
            self.scene_view.queue_refresh()

    def on_close(self):
        try:
            self.video_view.stop()
            self.pages_collection['view_page'].Enable()
            self.scene_view._view_roi = False
            self.scene_view.queue_refresh()
            self._enable_tool_scan(self.play_tool, False)
            self._enable_tool_scan(self.stop_tool, False)
            self._enable_tool_scan(self.pause_tool, False)
        except:
            pass

    def reset(self):
        self.video_view.reset()

    def setup_engine(self):
        self._enable_tool_scan(self.play_tool, True)
        self._enable_tool_scan(self.stop_tool, False)
        self._enable_tool_scan(self.pause_tool, False)

        driver.camera.read_profile()

        image_capture.texture_mode.read_profile('texture_scanning')
        image_capture.laser_mode.read_profile('laser_scanning')
        image_capture.set_remove_background(profile.settings['remove_background_scanning'])
        image_capture.set_mode_texture()

        laser_segmentation.read_profile('scanning')

        calibration_data.read_profile_camera()
        calibration_data.read_profile_calibration()

        ciclop_scan.read_profile()
        point_cloud_roi.read_profile()


    def get_image(self):
        if self.scanning:
            image_capture.stream = False
            image = current_video.capture()
            image = point_cloud_roi.mask_image(image)
            return image
        else:
            image_capture.stream = True
            image = image_capture.capture_texture()
            image = point_cloud_roi.draw_cross(image)
            if self.scene_view._view_roi:
                image = point_cloud_roi.mask_image(image)
                image = point_cloud_roi.draw_roi(image)
            return image

    def point_cloud_callback(self, range, progress, point_cloud, point_meta):
        point_cloud = point_cloud_roi.mask_point_cloud(*point_cloud)
        wx.CallAfter(self._point_cloud_callback,
                     range, progress, point_cloud, point_meta)

    def _point_cloud_callback(self, range, progress, point_cloud, meta):
        if range > 0:
            self.gauge.SetRange(range)
            self.gauge.SetValue(progress)
        if point_cloud is not None:
            (points, texture) = point_cloud
            self.scene_view.append_point_cloud(points, texture, meta = meta)

    def on_play_tool_clicked(self, event):
        if ciclop_scan._inactive:
            self._enable_tool_scan(self.play_tool, False)
            self._enable_tool_scan(self.pause_tool, True)
            ciclop_scan.resume()
        else:
            if not calibration_data.check_calibration():
                dlg = wx.MessageDialog(self,
                                       _("Calibration parameters are not correct.\n"
                                         "Please perform calibration process:\n"
                                         "  1. Scanner autocheck\n"
                                         "  2. Laser triangulation\n"
                                         "  3. Platform extrinsics"),
                                       _("Wrong calibration parameters"), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return

            if profile.settings['laser_triangulation_hash'] != calibration_data.md5_hash():
                dlg = wx.MessageDialog(self,
                                       _("Laser triangulation calibration has been performed \n"
                                         "with different camera intrinsics values.\n"
                                         "Please perform Laser triangulation calibration again:\n"
                                         "  1. Scanner autocheck\n"
                                         "  2. Laser triangulation"),
                                       _("Wrong calibration parameters"), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return

            if profile.settings['platform_extrinsics_hash'] != calibration_data.md5_hash():
                dlg = wx.MessageDialog(self,
                                       _("Platform extrinsics calibration has been performed \n"
                                         "with different camera intrinsics values.\n"
                                         "Please perform Platform extrinsics calibration again:\n"
                                         "  1. Scanner autocheck\n"
                                         "  2. Platform extrinsics"),
                                       _("Wrong calibration parameters"), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return

            if profile.settings['texture_mode'] == 'Laser BG' and profile.settings['remove_background_scanning'] == False:
                dlg = wx.MessageDialog(self,
                                       _("'Laser BG' texture mode selected but laser background removal disabled. Background will not be captured. \n"
                                         "You can enable laser background option at Adjustment->Scan capture->Lasers\n"
                                         "Scan with flat color background?\n"),
                                       _("Laser BG impossible"), wx.YES_NO | wx.ICON_QUESTION)
                result = dlg.ShowModal() == wx.ID_YES
                dlg.Destroy()
                if not result:
                    return

            result = True
            if self.scene_view._object is not None:
                dlg = wx.MessageDialog(self,
                                       _("Your current model will be deleted.\n"
                                         "Are you sure you want to delete it?"),
                                       _("Clear point cloud"), wx.YES_NO | wx.ICON_QUESTION)
                result = dlg.ShowModal() == wx.ID_YES
                dlg.Destroy()
            if result:
                ciclop_scan.set_callbacks(self.before_scan,
                                          None, lambda r: wx.CallAfter(self.after_scan, r))
                ciclop_scan.start()

    def before_scan(self):
        self.scene_view._view_roi = False
        self.scanning = True
        self._enable_tool_scan(self.play_tool, False)
        self._enable_tool_scan(self.stop_tool, True)
        self._enable_tool_scan(self.pause_tool, True)
        self.GetParent().enable_gui(False)
        self.scroll_panel.Hide()
        self.scroll_panel.GetParent().Layout()
        self.scroll_panel.Layout()
        self.GetParent().Layout()
        self.pages_collection['view_page'].combo_video_views.Show()
        self.scene_view.set_show_delete_menu(False)
        obj = self.scene_view.create_default_object()
        obj._mesh.metadata = {}
        meta_names = ['motor_step_scanning', 'texture_mode', 'use_laser', 'camera_matrix', 'distortion_vector',\
                'distance_left','normal_left','distance_right','normal_right',\
                'rotation_matrix', 'translation_vector']
        for n in meta_names:
            obj._mesh.metadata[n] = profile.settings[n]
        print("Metadata created: {0}".format(obj._mesh.metadata))
        self.gauge.SetValue(0)
        self.gauge.Show()
        self.scene_panel.Layout()
        self.Layout()

    def after_scan(self, response):
        ret, result = response
        if ret:
            self.gauge.SetValue(self.gauge.GetRange())
            dlg = wx.MessageDialog(self,
                                   _("Scanning has finished. If you want to save your "
                                     "point cloud go to \"File > Save model\""),
                                   _("Scanning finished!"), wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
            self.scanning = False
            # Flush video
            image_capture.capture_texture()
            image_capture.capture_texture()
            image_capture.capture_texture()
            self.on_scan_finished()
        else:
            if isinstance(result, InputOutputError):
                self.scanning = False
                self.on_scan_finished()
                self.GetParent().toolbar.update_status(False)
                driver.disconnect()
                dlg = wx.MessageDialog(
                    self,
                    "Low exposure values can cause a timing issue at the USB stack level on "
                    "v4l2_ioctl function in VIDIOC_S_CTRL mode. This is a Logitech issue on Linux",
                    str(result), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()

    def on_stop_tool_clicked(self, event):
        paused = ciclop_scan._inactive
        ciclop_scan.pause()
        dlg = wx.MessageDialog(self,
                               _("Your current scanning will be stopped.\n"
                                 "Are you sure you want to stop?"),
                               _("Stop scanning"), wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal() == wx.ID_YES
        dlg.Destroy()

        if result:
            self.scanning = False
            ciclop_scan.stop()
            # Flush video
            image_capture.capture_texture()
            image_capture.capture_texture()
            image_capture.capture_texture()
            self.on_scan_finished()
        else:
            if not paused:
                ciclop_scan.resume()

    def on_scan_finished(self):
        self._enable_tool_scan(self.play_tool, True)
        self._enable_tool_scan(self.stop_tool, False)
        self._enable_tool_scan(self.pause_tool, False)
        self.GetParent().enable_gui(True)
        self.GetParent().on_scanning_panel_clicked(None)
        self.pages_collection['view_page'].combo_video_views.Hide()
        self.scene_view.set_show_delete_menu(True)
        if profile.settings['current_panel_scanning'] == 'point_cloud_roi':
            self.scene_view._view_roi = profile.settings['use_roi']
            self.scene_view.queue_refresh()
        self.gauge.SetValue(0)
        self.gauge.Hide()
        self.scene_panel.Layout()
        self.Layout()

    def on_pause_tool_clicked(self, event):
        self._enable_tool_scan(self.play_tool, True)
        self._enable_tool_scan(self.pause_tool, False)
        ciclop_scan.pause()

    def _enable_tool_scan(self, item, enable):
        self.toolbar_scan.EnableTool(item.GetId(), enable)
