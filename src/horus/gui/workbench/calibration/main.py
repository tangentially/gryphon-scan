# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

from horus.util import profile

from horus.gui.engine import driver, pattern, calibration_data, image_capture, \
    image_detection, laser_segmentation, laser_triangulation, platform_extrinsics, \
    combo_calibration
from horus.gui.util.video_view import VideoView
from horus.gui.workbench.workbench import Workbench
from horus.gui.workbench.calibration.panels import PatternSettings, CameraIntrinsics, \
    ScannerAutocheck, RotatingPlatform, LaserTriangulation, PlatformExtrinsics, VideoSettings

from horus.gui.workbench.calibration.pages.camera_intrinsics import CameraIntrinsicsPages
from horus.gui.workbench.calibration.pages.scanner_autocheck import ScannerAutocheckPages
from horus.gui.workbench.calibration.pages.laser_triangulation import LaserTriangulationPages
from horus.gui.workbench.calibration.pages.platform_extrinsics import PlatformExtrinsicsPages
from horus.gui.workbench.calibration.pages.pattern_settings import PatternSettingsPages


class CalibrationWorkbench(Workbench):

    def __init__(self, parent):
        self.engine_mode = 'calibration'
        Workbench.__init__(self, parent, name=_('Calibration workbench'))

    def add_panels(self):
        self.add_panel(
            'pattern_settings', PatternSettings,
            self.on_pattern_settings_selected)
        self.add_panel(
            'scanner_autocheck', ScannerAutocheck,
            self.on_scanner_autocheck_selected)
        self.add_panel(
            'rotating_platform_settings', RotatingPlatform,
            self.on_rotating_platform_settings_selected)
        self.add_panel(
            'laser_triangulation', LaserTriangulation,
            self.on_laser_triangulation_selected)
        self.add_panel(
            'platform_extrinsics', PlatformExtrinsics,
            self.on_platform_extrinsics_selected)
        self.add_panel(
            'video_settings', VideoSettings,
            self.on_video_settings_selected)
        self.add_panel(
            'camera_intrinsics', CameraIntrinsics,
            self.on_camera_intrinsics_selected)

    def add_pages(self):
        self.add_page('video_view', VideoView(self, self.get_image))
        self.add_page('camera_intrinsics_pages', CameraIntrinsicsPages(
            self, start_callback=self.disable_panels, exit_callback=self.update_panels))
        self.add_page('scanner_autocheck_pages', ScannerAutocheckPages(
            self, start_callback=self.disable_panels, exit_callback=self.update_panels))
        self.add_page('laser_triangulation_pages', LaserTriangulationPages(
            self, start_callback=self.disable_panels, exit_callback=self.update_panels))
        self.add_page('platform_extrinsics_pages', PlatformExtrinsicsPages(
            self, start_callback=self.disable_panels, exit_callback=self.update_panels))
        self.add_page('pattern_settings_pages', PatternSettingsPages(self))

        self.pages_collection['camera_intrinsics_pages'].Hide()
        self.pages_collection['scanner_autocheck_pages'].Hide()
        self.pages_collection['laser_triangulation_pages'].Hide()
        self.pages_collection['platform_extrinsics_pages'].Hide()
        self.pages_collection['pattern_settings_pages'].Hide()

        self.pages_collection['camera_intrinsics_pages'].Disable()
        self.pages_collection['scanner_autocheck_pages'].Disable()
        self.pages_collection['laser_triangulation_pages'].Disable()
        self.pages_collection['platform_extrinsics_pages'].Disable()
        self.pages_collection['pattern_settings_pages'].Disable()

        if not profile.settings['view_mode_advanced']:
            self.panels_collection.expandable_panels['video_settings'].Hide()
            self.panels_collection.expandable_panels['camera_intrinsics'].Hide()

            if profile.settings['current_panel_calibration'] == 'video_settings' or \
               profile.settings['current_panel_calibration'] == 'camera_intrinsics':
                self.on_pattern_settings_selected()

        if profile.settings['view_hide_help']:
            #self.pages_collection['video_view'].info_panel.Hide()
            #self.pages_collection['camera_intrinsics_pages'].info_panel.Hide()
            self.pages_collection['scanner_autocheck_pages'].video_page.info_panel.Hide()
            self.pages_collection['laser_triangulation_pages'].video_page.info_panel.Hide()
            self.pages_collection['platform_extrinsics_pages'].video_page.info_panel.Hide()
            self.pages_collection['pattern_settings_pages'].info_panel.Hide()

        self.panels_collection.expandable_panels[
            profile.settings['current_panel_calibration']].on_title_clicked(None)

    def get_image(self):
        image = image_capture.capture_pattern()
        return image_detection.detect_pattern(image)

    def on_open(self):
        if driver.is_connected:
            self.pages_collection['camera_intrinsics_pages'].Enable()
            self.pages_collection['scanner_autocheck_pages'].Enable()
            self.pages_collection['laser_triangulation_pages'].Enable()
            self.pages_collection['platform_extrinsics_pages'].Enable()
            self.pages_collection['pattern_settings_pages'].Enable()
        else:
            for page in self.pages_collection:
                self.pages_collection[page].stop()
            self.pages_collection['camera_intrinsics_pages'].Disable()
            self.pages_collection['scanner_autocheck_pages'].Disable()
            self.pages_collection['laser_triangulation_pages'].Disable()
            self.pages_collection['platform_extrinsics_pages'].Disable()
            self.pages_collection['pattern_settings_pages'].Disable()

        self.panels_collection.expandable_panels[
            profile.settings['current_panel_calibration']].on_title_clicked(None)

    def on_close(self):
        try:
            for page in self.pages_collection:
                self.pages_collection[page].stop()
        except:
            pass

    def reset(self):
        for page in self.pages_collection:
            self.pages_collection[page].reset()

    def setup_engine(self):
        self.engine_mode = 'calibration'

        driver.camera.read_profile()
        image_capture.pattern_mode.read_profile('pattern_calibration')
        image_capture.texture_mode.read_profile('texture_scanning')
        image_capture.laser_mode.read_profile('laser_calibration')

        image_capture.set_remove_background(profile.settings['remove_background_calibration'])

        laser_segmentation.read_profile('calibration')
        pattern.read_profile()

        calibration_data.read_profile_camera()

        laser_triangulation.read_profile()
        platform_extrinsics.read_profile()
        combo_calibration.read_profile()

        image_capture.set_mode_pattern()

    def switch_engine_mode(self, mode='calibration'): # 'scanning'
        if self.engine_mode == mode:
            return
    
        engine_mode = mode

        image_capture.laser_mode.read_profile('laser_'+mode)
        image_capture.set_remove_background(profile.settings['remove_background_'+mode])
    
        laser_segmentation.read_profile(mode)

    def on_pattern_settings_selected(self):
        profile.settings['current_panel_calibration'] = 'pattern_settings'
        self.switch_engine_mode('calibration')
        self._on_panel_selected(self.pages_collection['pattern_settings_pages'])

    def on_rotating_platform_settings_selected(self):
        profile.settings['current_panel_calibration'] = 'rotating_platform_settings'
        self.switch_engine_mode('calibration')
        self._on_panel_selected(self.pages_collection['video_view'])

    def on_video_settings_selected(self):
        profile.settings['current_panel_calibration'] = 'video_settings'
        self.switch_engine_mode('calibration')
        self._on_panel_selected(self.pages_collection['video_view'])

    def on_camera_intrinsics_selected(self):
        profile.settings['current_panel_calibration'] = 'camera_intrinsics'
        self.switch_engine_mode('calibration')
        self._on_panel_selected(self.pages_collection['camera_intrinsics_pages'])

    def on_scanner_autocheck_selected(self):
        profile.settings['current_panel_calibration'] = 'scanner_autocheck'
        self.switch_engine_mode('calibration')
        self._on_panel_selected(self.pages_collection['scanner_autocheck_pages'])

    def on_laser_triangulation_selected(self):
        profile.settings['current_panel_calibration'] = 'laser_triangulation'
        self.switch_engine_mode('calibration')
        self._on_panel_selected(self.pages_collection['laser_triangulation_pages'])

    def on_platform_extrinsics_selected(self):
        profile.settings['current_panel_calibration'] = 'platform_extrinsics'
        self.switch_engine_mode('calibration')
        self._on_panel_selected(self.pages_collection['platform_extrinsics_pages'])

    def disable_panels(self):
        self.GetParent().enable_gui(False)
        self.scroll_panel.Disable()

    def update_panels(self):
        self.update_controls()
        self.GetParent().enable_gui(True)
        self.scroll_panel.Enable()

    def _on_panel_selected(self, panel):
        for page in self.pages_collection:
            self.pages_collection[page].Hide()
            self.pages_collection[page].stop()
        panel.Show()
        if driver.is_connected:
            panel.play()
        self.Layout()
