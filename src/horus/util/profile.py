# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>\
              Nicanor Romero Venier <nicanor.romerovenier@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.\
                 Copyright (C) 2013 David Braam from Cura Project'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import os
import math
import sys
import collections.abc
import json
import numpy as np

from horus import Singleton
import logging
logger = logging.getLogger(__name__)

from horus.util import resources, system


@Singleton
class Settings(collections.abc.MutableMapping):

    def __init__(self):
        self._settings_dict = dict()
        self.settings_version = 1

    # Getters

    def __getitem__(self, key):
        # For convinience, this returns the Setting value and not the Setting object itself
        value = self.get_setting(key).value
        if value is not None:
            return value
        else:
            return self.get_default(key)

    def setting_exists(self, key):
        if key in self._settings_dict:
            return True
        else:
            return False

    def get_setting(self, key):
        if self.setting_exists(key):
            return self._settings_dict[key]
        else:
            return None

    def get_label(self, key):
        if self.setting_exists(key):
            return self.get_setting(key)._label
        else:
            return ''

    def get_default(self, key):
        if self.get_setting(key)._type == np.ndarray:
            return self.get_setting(key).default.copy()
        else:
            return self.get_setting(key).default

    def get_min_value(self, key):
        return self.get_setting(key).min_value

    def get_max_value(self, key):
        return self.get_setting(key).max_value

    def get_possible_values(self, key):
        return self.get_setting(key)._possible_values

    # Setters

    def __setitem__(self, key, value):
        # For convinience, this sets the Setting value and not a Setting object
        self.cast_and_set(key, value)

    def set_min_value(self, key, value):
        self.get_setting(key).min_value = value

    def set_max_value(self, key, value):
        self.get_setting(key).max_value = value

    def cast_and_set(self, key, value):
        # if len(value) == 0:
        #    return
        setting_type = self.get_setting(key)._type
        try:
            if setting_type == bool:
                value = bool(value)
            elif setting_type == int:
                value = int(value)
            elif setting_type == float:
                value = float(value)
            elif setting_type == str:
                value = str(value)
            elif setting_type == list:
                value = value
            elif setting_type == np.ndarray:
                value = np.asarray(value)
        except:
            logger.error("Unable to cast setting %s to type %s" % (key, setting_type))
        else:
            self.get_setting(key).value = value

    # File management

    def load_settings(self, filepath=None, categories=None):
        if filepath is None:
            filepath = os.path.join(get_base_path(), 'settings.json')
        with open(filepath, 'r') as f:
            self._load_json_dict(json.loads(f.read()), categories)

    def _load_json_dict(self, json_dict, categories):
        for category in json_dict.keys():
            if category == "settings_version":
                continue
            if categories is None or category in categories:
                for key in json_dict[category]:
                    if key in self._settings_dict:
                        self._convert_to_type(key, json_dict[category][key])
                        self.get_setting(key)._load_json_dict(json_dict[category][key])

    def _convert_to_type(self, key, json_dict):
        if self._settings_dict[key]._type == np.ndarray:
            json_dict['value'] = np.asarray(json_dict['value'])

    def save_settings(self, filepath=None, categories=None):
        if filepath is None:
            filepath = os.path.join(get_base_path(), 'settings.json')

        # If trying to overwrite some categories of settings.json, first load it
        # to preserve the other values
        if categories is not None and filepath == os.path.join(get_base_path(), 'settings.json'):
            with open(filepath, 'r') as f:
                initial_json = json.loads(f.read())
        else:
            initial_json = None

        with open(filepath, 'w') as f:
            f.write(
                json.dumps(self._to_json_dict(categories, initial_json), sort_keys=True, indent=4))
        logger.info("Saving settings to {0}".format(filepath))

    def _to_json_dict(self, categories, initial_json=None):
        if initial_json is None:
            json_dict = dict()
        else:
            json_dict = initial_json.copy()

        json_dict["settings_version"] = self.settings_version
        for key in self._settings_dict.keys():
            cur_setting = self.get_setting(key)
            if categories is not None and cur_setting._category not in categories:
                continue
            # hack to use panel add_control()
            if cur_setting._category == 'no_settings':
                continue
            if cur_setting._category not in json_dict:
                json_dict[cur_setting._category] = dict()
                #print "Profile category {0}".format(cur_setting._category)
            json_dict[cur_setting._category][key] = cur_setting._to_json_dict()
        return json_dict

    # Other

    def __delitem__(self, key):
        del self._settings_dict[key]

    def __iter__(self):
        return iter(self._settings_dict)

    def __len__(self):
        return len(self._settings_dict)

    def reset_to_default(self, key=None, categories=None):
        if key is not None:
            self.__setitem__(key, self.get_default(key))
        else:
            for key in self._settings_dict.keys():
                if categories is not None and self.get_setting(key)._category not in categories:
                    continue
                self.__setitem__(key, self.get_default(key))

    def _add_setting(self, setting):
        if not setting._id in self._settings_dict:
            self._settings_dict[setting._id] = setting

    def _initialize_settings(self):

        # ============== CONNECTION Preferences ========

        self._add_setting(
            Setting('serial_name', _('Serial name'), 'preferences', str, ''))
        self._add_setting(
            Setting('baud_rate', _('Baud rate'), 'preferences', int, 115200,
                    possible_values=(9600, 14400, 19200, 38400, 57600, 115200)))
        self._add_setting(
            Setting('camera_id', _('Camera ID'), 'preferences', str, ''))
        self._add_setting(
            Setting('board', _('Board'), 'preferences', str, 'BT ATmega328',
                    possible_values=('Arduino Uno', 'BT ATmega328')))
        self._add_setting(
            Setting('firmware_string', 'Firmware version string', 'preferences', str, "Horus 0.2 ['$' for help]"))
        self._add_setting(
            Setting('init_string', 'Board init string', 'preferences', str, ''))
        self._add_setting(
            Setting('invert_motor', _('Invert motor'), 'preferences', bool, False))
        self._add_setting(
            Setting('language', _('Language'), 'preferences', str, 'English',
                    possible_values=('English', 'Español', 'Français',
                                     'Deutsch', 'Italiano', 'Português'),
                    tooltip=_('Change the language of Horus. '
                              'Switching language will require a program restart')))


        # ========== Camera profiles ===========
        # Hack to translate combo boxes:
        _('Very high')
        _('High')
        _('Medium')
        _('Low')

        # ------- Control -----------
        self._add_setting(
            Setting('luminosity', _('Luminosity'), 'profile_settings',
                    str, 'Medium', possible_values=('High', 'Medium', 'Low')))
        self._add_setting(
            Setting('brightness_control', _('Brightness'), 'profile_settings',
                    int, 128, min_value=0, max_value=255))
        self._add_setting(
            Setting('contrast_control', _('Contrast'), 'profile_settings',
                    int, 32, min_value=0, max_value=255))
        self._add_setting(
            Setting('saturation_control', _('Saturation'), 'profile_settings',
                    int, 32, min_value=0, max_value=255))
        self._add_setting(
            Setting('exposure_control', _('Exposure'), 'profile_settings',
                    int, 16, min_value=1, max_value=64))
        self._add_setting(
            Setting('light1_control', _('Lamp 1 brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))
        self._add_setting(
            Setting('light2_control', _('Lamp 2 brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))

        # -------- Calibration --------
        self._add_setting(
            Setting('brightness_pattern_calibration', _('Brightness'), 'profile_settings',
                    int, 128, min_value=0, max_value=255))
        self._add_setting(
            Setting('contrast_pattern_calibration', _('Contrast'), 'profile_settings',
                    int, 32, min_value=0, max_value=255))
        self._add_setting(
            Setting('saturation_pattern_calibration', _('Saturation'), 'profile_settings',
                    int, 32, min_value=0, max_value=255))
        self._add_setting(
            Setting('exposure_pattern_calibration', _('Exposure'), 'profile_settings',
                    int, 16, min_value=1, max_value=64))
        self._add_setting(
            Setting('light1_pattern_calibration', _('Lamp 1 brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))
        self._add_setting(
            Setting('light2_pattern_calibration', _('Lamp 2 brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))

        self._add_setting(
            Setting('brightness_laser_calibration', _('Brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))
        self._add_setting(
            Setting('contrast_laser_calibration', _('Contrast'), 'profile_settings',
                    int, 100, min_value=0, max_value=255))
        self._add_setting(
            Setting('saturation_laser_calibration', _('Saturation'), 'profile_settings',
                    int, 100, min_value=0, max_value=255))
        self._add_setting(
            Setting('exposure_laser_calibration', _('Exposure'), 'profile_settings',
                    int, 8, min_value=1, max_value=64))
        self._add_setting(
            Setting('light1_laser_calibration', _('Lamp 1 brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))
        self._add_setting(
            Setting('light2_laser_calibration', _('Lamp 2 brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))

        self._add_setting(
            Setting('remove_background_calibration', _('Remove background'),
                    'profile_settings', bool, True))

        # ------- Scanning ------
        self._add_setting(
            Setting('brightness_texture_scanning', _('Brightness'), 'profile_settings',
                    int, 128, min_value=0, max_value=255))
        self._add_setting(
            Setting('contrast_texture_scanning', _('Contrast'), 'profile_settings',
                    int, 32, min_value=0, max_value=255))
        self._add_setting(
            Setting('saturation_texture_scanning', _('Saturation'), 'profile_settings',
                    int, 50, min_value=0, max_value=255))
        self._add_setting(
            Setting('exposure_texture_scanning', _('Exposure'), 'profile_settings',
                    int, 16, min_value=1, max_value=64))
        self._add_setting(
            Setting('light1_texture_scanning', _('Lamp 1 brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))
        self._add_setting(
            Setting('light2_texture_scanning', _('Lamp 2 brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))


        self._add_setting(
            Setting('brightness_laser_scanning', _('Brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))
        self._add_setting(
            Setting('contrast_laser_scanning', _('Contrast'), 'profile_settings',
                    int, 100, min_value=0, max_value=255))
        self._add_setting(
            Setting('saturation_laser_scanning', _('Saturation'), 'profile_settings',
                    int, 100, min_value=0, max_value=255))
        self._add_setting(
            Setting('exposure_laser_scanning', _('Exposure'), 'profile_settings',
                    int, 8, min_value=1, max_value=64))
        self._add_setting(
            Setting('light1_laser_scanning', _('Lamp 1 brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))
        self._add_setting(
            Setting('light2_laser_scanning', _('Lamp 2 brightness'), 'profile_settings',
                    int, 0, min_value=0, max_value=255))

        self._add_setting(
            Setting('remove_background_scanning', _('Remove background'),
                    'profile_settings', bool, True))


        # ============ Video flush profiles ==============
        # [ texture, laser, pattern, change mode ]
        # - Linux
        self._add_setting(
            Setting('flush_linux', 'Flush Linux', 'preferences',
                    np.ndarray, np.ndarray(shape=(4,), dtype=int, buffer=np.array([3, 2, 3, 0]))))
        self._add_setting(
            Setting('flush_stream_linux', 'Flush stream Linux', 'preferences',
                    np.ndarray, np.ndarray(shape=(4,), dtype=int, buffer=np.array([0, 3, 3, 0]))))
        # - Darwin
        self._add_setting(
            Setting('flush_darwin', 'Flush Darwin', 'preferences',
                    np.ndarray, np.ndarray(shape=(4,), dtype=int, buffer=np.array([4, 3, 4, 0]))))
        self._add_setting(
            Setting('flush_stream_darwin', 'Flush stream Darwin', 'preferences',
                    np.ndarray, np.ndarray(shape=(4,), dtype=int, buffer=np.array([0, 3, 3, 0]))))
        # - Windows
        self._add_setting(
            Setting('flush_windows', 'Flush Windows', 'preferences',
                    np.ndarray, np.ndarray(shape=(4,), dtype=int, buffer=np.array([4, 3, 4, 0]))))
        self._add_setting(
            Setting('flush_stream_windows', 'Flush stream Windows', 'preferences',
                    np.ndarray, np.ndarray(shape=(4,), dtype=int, buffer=np.array([0, 3, 3, 0]))))


        # ========== Segmentation profiles ===========

        # -------- Calibration --------
        self._add_setting(
            Setting('laser_color_detector_calibration', _('Laser color detector'), 'profile_settings',
                    str, 'R (HSV)',
                    possible_values=('R (RGB)', 'G (RGB)', 'B (RGB)', 'R (HSV)', 'Cr (YCrCb)', 'U (YUV)')))
        self._add_setting(
            Setting('threshold_enable_calibration', _('Enable threshold'),
                    'profile_settings', bool, True))
        self._add_setting(
            Setting('threshold_value_calibration', _('Threshold'), 'profile_settings',
                    int, 50, min_value=0, max_value=255))
        self._add_setting(
            Setting('blur_enable_calibration', _('Enable blur'),
                    'profile_settings', bool, False))
        self._add_setting(
            Setting('blur_value_calibration', _('Blur'), 'profile_settings',
                    int, 2, min_value=0, max_value=10))
        self._add_setting(
            Setting('window_enable_calibration', _('Enable window'),
                    'profile_settings', bool, True))
        self._add_setting(
            Setting('window_value_calibration', _('Window'), 'profile_settings',
                    int, 5, min_value=0, max_value=30))
        self._add_setting(
            Setting('refinement_calibration', _('Refinement'), 'profile_settings',
                    str, 'RANSAC',
                    possible_values=('None', 'SGF', 'RANSAC')))

        # -------- Scanning --------
        self._add_setting(
            Setting('laser_color_detector_scanning', _('Laser color detector'), 'profile_settings',
                    str, 'R (HSV)',
                    possible_values=('R (RGB)', 'G (RGB)', 'B (RGB)', 'R (HSV)', 'Cr (YCrCb)', 'U (YUV)')))
        self._add_setting(
            Setting('threshold_enable_scanning', _('Enable threshold'),
                    'profile_settings', bool, True))
        self._add_setting(
            Setting('threshold_value_scanning', _('Threshold'), 'profile_settings',
                    int, 50, min_value=0, max_value=255))
        self._add_setting(
            Setting('blur_enable_scanning', _('Enable blur'),
                    'profile_settings', bool, True))
        self._add_setting(
            Setting('blur_value_scanning', _('Blur'), 'profile_settings',
                    int, 2, min_value=0, max_value=10))
        self._add_setting(
            Setting('window_enable_scanning', _('Enable window'),
                    'profile_settings', bool, True))
        self._add_setting(
            Setting('window_value_scanning', _('Window'), 'profile_settings',
                    int, 8, min_value=0, max_value=30))
        self._add_setting(
            Setting('refinement_scanning', _('Refinement'), 'profile_settings',
                    str, 'SGF',
                    possible_values=('None', 'SGF')))


        # ==================== CONTROL workbench ================

        self._add_setting(
            Setting('current_panel_control', 'camera_control', 'profile_settings',
                    str, 'camera_control',
                    possible_values=('camera_control', 'laser_control',
                                     'ldr_value', 'motor_control', 'gcode_control')))

        self._add_setting(
            Setting('frame_rate', _('Frame rate'), 'profile_settings',
                    int, 30, possible_values=(30, 25, 20, 15, 10, 5)))

        self._add_setting(
            Setting('motor_step_control', _('Step (º)'), 'profile_settings',
                    float, 90.0))
        self._add_setting(
            Setting('motor_speed_control', _('Speed (º/s)'), 'profile_settings',
                    float, 200.0, min_value=1.0, max_value=1000.0))
        self._add_setting(
            Setting('motor_acceleration_control', _('Acceleration (º/s²)'), 'profile_settings',
                    float, 200.0, min_value=1.0, max_value=1000.0))


        self._add_setting(
            Setting('save_image_button', _('Save image'), 'no_settings', str, ''))
        self._add_setting(
            Setting('left_button', _('Left'), 'no_settings', str, ''))
        self._add_setting(
            Setting('right_button', _('Right'), 'no_settings', str, ''))
        self._add_setting(
            Setting('move_button', _('Move'), 'no_settings', str, ''))
        self._add_setting(
            Setting('enable_button', _('Enable'), 'no_settings', str, ''))
        self._add_setting(
            Setting('reset_origin_button', _('Reset origin'), 'no_settings', str, ''))
        self._add_setting(
            Setting('gcode_gui', _('Send'), 'no_settings', str, ''))
        self._add_setting(
            Setting('ldr_value', _('Send'), 'no_settings', str, ''))



        # ==================== ADJUSTMENT workbench ================

        self._add_setting(
            Setting('current_panel_adjustment', 'scan_capture', 'profile_settings',
                    str, 'scan_capture',
                    possible_values=('scan_capture', 'scan_segmentation',
                                     'calibration_capture', 'calibration_segmentation')))

        self._add_setting(
            Setting('current_video_mode_adjustment', 'Texture', 'profile_settings',
                    str, 'Texture',
                    possible_values=('Texture', 'Pattern', 'Laser', 'Gray')))

        self._add_setting(
            Setting('capture_mode_scanning', _('Capture mode'), 'profile_settings',
                    str, 'Texture', possible_values=('Texture', 'Laser')))
        self._add_setting(
            Setting('draw_line_scanning', _('Draw line'), 'profile_settings', bool, True))

        self._add_setting(
            Setting('capture_mode_calibration', _('Capture mode'), 'profile_settings',
                    str, 'Pattern', possible_values=('Pattern', 'Laser')))
        self._add_setting(
            Setting('draw_line_calibration', _('Draw line'), 'profile_settings', bool, True))



        # ==================== CALIBRATION workbench ================

        self._add_setting(
            Setting('current_panel_calibration', 'pattern_settings', 'profile_settings',
                    str, 'pattern_settings',
                    possible_values=('pattern_settings', 'camera_intrinsics',
                                     'scanner_autocheck', 'rotating_platform_settings',
                                     'laser_triangulation', 'platform_extrinsics',
                                     'video_settings'))) #, 'cloud_correction')))

        # ----- Pattern Settings ---------
        self._add_setting(
            Setting('pattern_rows', _('Pattern rows'), 'calibration_settings',
                    int, 6, min_value=2, max_value=50))
        self._add_setting(
            Setting('pattern_columns', _('Pattern columns'), 'calibration_settings',
                    int, 11, min_value=2, max_value=50))
        self._add_setting(
            Setting('pattern_square_width', _('Square width (mm)'), 'calibration_settings',
                    float, 13.0, min_value=1.0))
        self._add_setting(
            Setting('pattern_origin_distance', _('Origin distance (mm)'), 'calibration_settings',
                    float, 0.0, min_value=0.0))
        self._add_setting(
            Setting('pattern_border_l', _('Border Left (mm)'), 'calibration_settings',
                    float, 5.0, min_value=0.0))
        self._add_setting(
            Setting('pattern_border_r', _('Border Right (mm)'), 'calibration_settings',
                    float, 5.0, min_value=0.0))
        self._add_setting(
            Setting('pattern_border_t', _('Border Top (mm)'), 'calibration_settings',
                    float, 5.0, min_value=0.0))
        self._add_setting(
            Setting('pattern_border_b', _('Border Bottom (mm)'), 'calibration_settings',
                    float, 5.0, min_value=0.0))

        # ----- Scanner Autocheck ---------
        self._add_setting(
            Setting('autocheck_button', _('Perform autocheck'), 'no_settings', str, ''))

        # ----- Rotating Platform ---------
        self._add_setting(
            Setting('motor_step_calibration', _('Step (º)'), 'calibration_settings',
                    float, 4.5))
        self._add_setting(
            Setting('motor_speed_calibration', _('Speed (º/s)'), 'calibration_settings',
                    float, 200.0, min_value=1.0, max_value=1000.0))
        self._add_setting(
            Setting('motor_acceleration_calibration', _('Acceleration (º/s²)'),
                    'calibration_settings', float, 200.0, min_value=1.0, max_value=1000.0))

        self._add_setting(
            Setting('after_calibration_position', _('Platform position after calibration'), 'calibration_settings',
                    str, 'Return', possible_values=('Keep', 'Return', 'Perpendicular')))

        # ----- Laser Triangulation ---------
        self._add_setting(
            Setting('distance_left', _('Distance left (mm)'), 'calibration_settings', float, 0.0))
        self._add_setting(
            Setting('normal_left', _('Normal left'), 'calibration_settings',
                    np.ndarray, np.ndarray(shape=(3,), buffer=np.array([0.0, 0.0, 0.0]))))

        self._add_setting(
            Setting('distance_right', _('Distance right (mm)'), 'calibration_settings', float, 0.0))
        self._add_setting(
            Setting('normal_right', _('Normal right'), 'calibration_settings',
                    np.ndarray, np.ndarray(shape=(3,), buffer=np.array([0.0, 0.0, 0.0]))))

        self._add_setting(
            Setting('laser_triangulation_hash', '', 'calibration_settings', str, ''))

        self._add_setting(
            Setting('laser_calibration_angles', _('Laser on pattern visibility angle ranges'), 'calibration_settings',
                    np.ndarray, np.ndarray(shape=(2, 2), buffer=np.array([[-90.0, 90.0],
                                                                          [-90.0, 90.0]]))))

        # ----- Platform extrinsics ---------
        self._add_setting(
            Setting('rotation_matrix', _('Rotation matrix'), 'calibration_settings',
                    np.ndarray, np.ndarray(shape=(3, 3), buffer=np.array([[0.0, 0.0, 0.0],
                                                                          [0.0, 0.0, 0.0],
                                                                          [0.0, 0.0, 0.0]]))))
        self._add_setting(
            Setting('translation_vector', _('Translation vector (mm)'), 'calibration_settings',
                    np.ndarray, np.ndarray(shape=(3,), buffer=np.array([0.0, 0.0, 0.0]))))

        self._add_setting(
            Setting('platform_extrinsics_hash', '', 'calibration_settings', str, ''))

        # ----- Video settings ---------

        # some cameras like "Lenovo EasyCamera" require to grab frame before update settings
        self._add_setting(
            Setting('camera_capture_before_set', _('Capture a frame right after connect before setting camera'), 'profile_settings', bool, False))

        self._add_setting(
            Setting('camera_width', _('Width'), 'calibration_settings',
                    int, -1, min_value=-1, max_value=10000))
        self._add_setting(
            Setting('camera_height', _('Height'), 'calibration_settings',
                    int, -1, min_value=-1, max_value=10000))

        self._add_setting(
            Setting('camera_focus', _('Manual focus'), 'calibration_settings',
                    int, 0, min_value=0, max_value=255))

        self._add_setting(
            Setting('camera_rotate', _('Rotate'), 'calibration_settings', bool, True))
        self._add_setting(
            Setting('camera_hflip', _('Horizontal flip'), 'calibration_settings', bool, True))
        self._add_setting(
            Setting('camera_vflip', _('Vertical flip'), 'calibration_settings', bool, False))
        self._add_setting(
            Setting('set_resolution_button', _('Set resolution'), 'no_settings', str, ''))
        self._add_setting(
            Setting('auto_resolution', _('Use MAX resolution'), 'no_settings', bool, False))

        # ----- Camera intrinsics ---------
        self._add_setting(
            Setting('camera_matrix', _('Camera matrix'), 'calibration_settings',
                    np.ndarray, np.ndarray(shape=(3, 3), buffer=np.array([[1430.0, 0.0, 480.0],
                                                                          [0.0, 1430.0, 640.0],
                                                                          [0.0, 0.0, 1.0]]))))
        self._add_setting(
            Setting('distortion_vector', _('Distortion vector'), 'calibration_settings',
                    np.ndarray, np.ndarray(shape=(5,),
                                           buffer=np.array([0.0, 0.0, 0.0, 0.0, 0.0]))))

        # new camera calculator
        self._add_setting(
            Setting('new_camera_matrix', _('Initial new camera matrix'), 'no_settings',
                    np.ndarray, np.ndarray(shape=(3, 3), buffer=np.array([[0.0, 0.0, 0.0],
                                                                          [0.0, 0.0, 0.0],
                                                                          [0.0, 0.0, 0.0]]))))
        self._add_setting(
            Setting('new_camera_ruler', _('Target length (mm)'), 'no_settings', float, 325.0))
        self._add_setting(
            Setting('new_camera_distance_h', _('Target horizontal distance (mm)'), 'no_settings', float, 480.0))
        self._add_setting(
            Setting('new_camera_distance_v', _('Target vertical distance (mm)'), 'no_settings', float, 265.0))
        self._add_setting(
            Setting('apply_new_camera_button', _('Apply calculated camera data'), 'no_settings', str, ''))


        # ==================== SCANNING workbench ================

        self._add_setting(
            Setting('current_panel_scanning', 'scan_parameters', 'profile_settings',
                    str, 'scan_parameters',
                    possible_values=('scan_parameters', 'rotating_platform',
                                     'point_cloud_roi', 'point_cloud_color',
                                     'photogrammetry', 'mesh_correction')))

        self._add_setting(
            Setting('view_scanning_panel', _('View scanning panel'), 'preferences', bool, False))
        self._add_setting(
            Setting('view_scanning_video', _('View scanning video'), 'preferences', bool, False))
        self._add_setting(
            Setting('view_scanning_scene', _('View scanning scene'), 'preferences', bool, True))

        # Hack to translate combo boxes:
        _('Open')
        _('Enable open')

        # Hack to translate combo boxes:
        _('Texture')
        _('Laser')
        _('Gray')
        _('Line')
        self._add_setting(
            Setting('video_scanning', _('Video'), 'profile_settings',
                    str, 'Texture', possible_values=('Texture', 'Laser', 'Gray', 'Line')))

        # ----------- Scan parameters ----------
        # Hack to translate combo boxes:
        _('Left')
        _('Right')
        _('Both')
        self._add_setting(
            Setting('use_laser', _('Use laser'), 'profile_settings',
                    str, 'Both', possible_values=('Left', 'Right', 'Both')))

        # ----------- Rotating platform ----------
        self._add_setting(
            Setting('motor_step_scanning', _('Step (º)'), 'profile_settings',
                    float, 0.45))
        self._add_setting(
            Setting('motor_speed_scanning', _('Speed (º/s)'), 'profile_settings',
                    float, 200.0, min_value=1.0, max_value=1000.0))
        self._add_setting(
            Setting('motor_acceleration_scanning', _('Acceleration (º/s²)'), 'profile_settings',
                    float, 200.0, min_value=1.0, max_value=1000.0))

        # ----------- Point cloud ROI ----------
        self._add_setting(
            Setting('show_center', _('Show center'), 'profile_settings', bool, True))
        self._add_setting(
            Setting('use_roi', _('Use ROI'), 'profile_settings', bool, False))
        self._add_setting(
            Setting('roi_diameter', _('Diameter (mm)'), 'profile_settings',
                    int, 300, min_value=0, max_value=350))
        self._add_setting(
            Setting('roi_height', _('Height (mm)'), 'profile_settings',
                    int, 400, min_value=0, max_value=350))

        # ----------- Point cloud color ----------
        self._add_setting(
            Setting('texture_mode', _('Texture'), 'profile_settings', 
                    str, 'Texture',
                    possible_values=('Flat color', 'Multi color', 'Capture', 'Laser BG')))

        self._add_setting(
            Setting('point_cloud_color', _('Cloud color'), 'profile_settings',
                    list, [170,170,170]))
        self._add_setting(
            Setting('point_cloud_color_l', _('Left color'), 'profile_settings',
                    list, [255,0,0]))
        self._add_setting(
            Setting('point_cloud_color_r', _('Right color'), 'profile_settings',
                    list, [0,255,255]))

        # ------------- Photogrammetry ---------------
        self._add_setting(
            Setting('ph_save_enable', _('Save photos'), 'preferences', bool, False))
        self._add_setting(
            Setting('ph_save_folder', _('Images folder'), 'preferences', str, 'photo/' ))
        self._add_setting(
            Setting('ph_save_divider', 'Save every N\'th frame', 'preferences', int, 2, min_value=1, max_value=9999))


        # ------------- Mesh Correction ---------------
        self._add_setting(
            Setting('mesh_correction_offset', _('Center Offset (model space)'), 'no_settings',
                    np.ndarray, np.ndarray(shape=(3,), buffer=np.array([0.0, 0.0, 0.0]))))

        self._add_setting(
            Setting('mesh_correction_apply', _('Apply'), 'no_settings', str, ''))
        self._add_setting(
            Setting('mesh_correction_reset', _('Reset'), 'no_settings', str, ''))

        # ----------- Engine ----------
        self._add_setting(
            Setting('scan_sleep', _('Wait milliseconds after each scan capture step'), 'profile_settings',
                    float, 0.0, min_value=0.0, max_value=1000.0))

        self._add_setting(
            Setting('scan_sync_threads', _('Synchronize capture and process threads'),
                    'profile_settings', bool, False))



        # ========== MACHINE Profile ==============

        self._add_setting(
            Setting('machine_diameter', _('Machine diameter'), 'machine_settings', int, 304))
        self._add_setting(
            Setting('machine_width', _('Machine width'), 'machine_settings', int, 200))
        self._add_setting(
            Setting('machine_height', _('Machine height'), 'machine_settings', int, 200))
        self._add_setting(
            Setting('machine_depth', _('Machine depth'), 'machine_settings', int, 200))

        # Hack to translate combo boxes:
        _('Circular')
        _('Rectangular')
        self._add_setting(
            Setting('machine_shape', _('Machine shape'), 'machine_settings',
                    str, 'Circular', possible_values=('Circular', 'Rectangular')))
        self._add_setting(
            Setting('machine_model_path', _('Machine model'), 'machine_settings',
                    str, str(resources.get_path_for_mesh('Gryphon_platform.stl')))) # ciclop_platform.stl
        self._add_setting(
            Setting('machine_model_diameter', _('Machine model diameter (-1 dont scale; 0 auto scale)'), 'machine_settings', int, 304))
        self._add_setting(
            Setting('machine_model_offset_x', 'Machine model offset X', 'machine_settings',
                    float, 0.00))
        self._add_setting(
            Setting('machine_model_offset_y', 'Machine model offset Y', 'machine_settings',
                    float, 0.00))
        self._add_setting(
            Setting('machine_model_offset_z', 'Machine model offset Z', 'machine_settings',
                    float, 0.00)) # 8.05 for Ciclop
        self._add_setting(
            Setting('platform_border_z', 'Platform border draw z offset', 'machine_settings', int, 0))
        self._add_setting(
            Setting('platform_markers_diameter', 'Platform markers diameter', 'machine_settings', int, 276))
        self._add_setting(
            Setting('platform_markers_z', 'Platform markers draw z offset', 'machine_settings', int, 0))


        self._add_setting(
            Setting('point_size', 'Point size', 'preferences', int, 2, min_value=1, max_value=4))


        # =============== GUI General ================
        self._add_setting(
            Setting('workbench', _('Workbench'), 'preferences', str, 'scanning',
                    possible_values=('control', 'adjustment', 'calibration', 'scanning')))

        self._add_setting(
            Setting('show_welcome', _('Show welcome'), 'preferences', bool, True))
        self._add_setting(
            Setting('check_for_updates', _('Check for updates'), 'preferences', bool, True))

        #self._add_setting(
        #    Setting('basic_mode', _('Basic mode'), 'preferences', bool, False))
        #self._add_setting(
        #    Setting('view_control_panel', _('View control panel'), 'preferences', bool, True))
        #self._add_setting(
        #    Setting('view_control_video', _('View control panel'), 'preferences', bool, True))
        #self._add_setting(
        #    Setting('view_adjustment_panel', _('View adjustment panel'),
        #            'preferences', bool, True))
        #self._add_setting(
        #    Setting('view_adjustment_video', _('View adjustment video'),
        #            'preferences', bool, True))
        #self._add_setting(
        #    Setting('view_calibration_panel', _('View calibration panel'),
        #            'preferences', bool, True))
        #self._add_setting(
        #    Setting('view_calibration_video', _('View calibration video'),
        #            'preferences', bool, True))

        self._add_setting(
            Setting('view_mode_advanced', _('Advanced mode'), 'preferences', bool, True))
        self._add_setting(
            Setting('view_hide_help', _('Hide directions'), 'preferences', bool, False))

        self._add_setting(
            Setting('last_files', _('Last files'), 'preferences', list, []))
        # TODO: Set this default value
        self._add_setting(
            Setting('last_file', _('Last file'), 'preferences', str, ''))
        # TODO: Set this default value
        self._add_setting(
            Setting('last_profile', _('Last profile'), 'preferences', str, ''))
        self._add_setting(
            Setting('model_color', _('Default model color'), 'preferences', str, '888888'))
        self._add_setting(
            Setting('last_clear_log_date', _('Last clear log date'), 'preferences', str, ''))

        # wizard
        self._add_setting(
            Setting('adjust_laser', _('Adjust laser'), 'calibration_settings', bool, True))



class Setting(object):

    def __init__(self, setting_id, label, category, setting_type, default,
                 min_value=None, max_value=None, possible_values=None, tooltip='', tag=None):
        self._id = setting_id
        self._label = label
        self._category = category
        self._type = setting_type
        self._tooltip = tooltip
        self._tag = tag

        self.min_value = min_value
        self.max_value = max_value
        self._possible_values = possible_values
        self.default = default
        self.__value = None

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        if value is None:
            return
        self._check_type(value)
        value = self._check_range(value)
        value = self._check_possible_values(value)
        self.__value = value

    @property
    def default(self):
        return self.__default

    @default.setter
    def default(self, value):
        self._check_type(value)
        value = self._check_range(value)
        value = self._check_possible_values(value)
        self.__default = value

    @property
    def min_value(self):
        return self.__min_value

    @min_value.setter
    def min_value(self, value):
        if value is not None:
            self._check_type(value)
        self.__min_value = value

    @property
    def max_value(self):
        return self.__max_value

    @max_value.setter
    def max_value(self, value):
        if value is not None:
            self._check_type(value)
        self.__max_value = value

    def _check_type(self, value):
        if not isinstance(value, self._type):
            logger.error("Error when setting %s.\n%s (%s) is not of type %s. "
                         "Please remove current profile at ~/.horus" %
                         (self._id, value, type(value), self._type))

    def _check_range(self, value):
        if self.min_value is not None and value < self.min_value:
            logger.warning('Warning: For setting %s, %s is below min value %s.' % (self._id, value,
                           self.min_value))
            return self.min_value
        if self.max_value is not None and value > self.max_value:
            logger.warning('Warning: For setting %s.\n%s is above max value %s.' % (self._id, value,
                           self.max_value))
            return self.max_value
        return value

    def _check_possible_values(self, value):
        if self._possible_values is not None and value not in self._possible_values:
            logger.error('Error when setting %s.\n%s is not within the possible values %s.' % (
                self._id, value, self._possible_values))
            if len(self._possible_values) > 0:
                return self._possible_values[0]
        return value

    def _load_json_dict(self, json_dict):
        # Only load configurable fields (__value, __min_value, __max_value)
        self.value = json_dict['value']
        if 'min_value' in json_dict:
            self.min_value = json_dict['min_value']
        if 'max_value' in json_dict:
            self.max_value = json_dict['max_value']

    def _to_json_dict(self):
        # Convert only configurable fields
        json_dict = dict()

        if self.value is None:
            value = self.default
        else:
            value = self.value

        if self._type == np.ndarray and value is not None:
            json_dict['value'] = value.tolist()
        else:
            json_dict['value'] = value

        if self.min_value is not None:
            json_dict['min_value'] = self.min_value

        if self.max_value is not None:
            json_dict['max_value'] = self.max_value
        return json_dict


# Define a fake _() function to fake the gettext tools in to generating
# strings for the profile settings.

def _(n):
    return n

settings = Settings()
settings._initialize_settings()

# Remove fake defined _() because later the localization will define a global _()
del _


def get_base_path():
    """
    :return: The path in which the current configuration files are stored.
    This depends on the used OS.
    """
    if system.is_windows():
        basePath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        # If we have a frozen python install, we need to step out of the library.zip
        if hasattr(sys, 'frozen'):
            basePath = os.path.normpath(os.path.join(basePath, ".."))
    else:
        basePath = os.path.expanduser('~/.horus/')
    if not os.path.isdir(basePath):
        try:
            os.makedirs(basePath)
        except:
            logger.error("Failed to create directory: %s" % (basePath))
    return basePath


def load_settings():
    if os.path.exists(os.path.join(get_base_path(), 'settings.json')):
        settings.load_settings()
        return


# TODO: Move these somewhere else

# Returns a list of convex polygons, first polygon is the allowed area of the machine,
# the rest of the polygons are the dis-allowed areas of the machine.
def get_machine_size_polygons():
    machine_shape = settings["machine_shape"]
    if machine_shape == "Circular":
        size = np.array([ settings['machine_diameter'],
                          settings['machine_diameter'],
                          settings['machine_height'] ], np.float32)
    elif machine_shape == "Rectangular":
        size = np.array([ settings['machine_width'],
                          settings['machine_depth'],
                          settings['machine_height'] ], np.float32)
    return get_size_polygons(size, machine_shape)


def get_roi_size_polygons():
    machine_shape = settings["machine_shape"]
    if machine_shape == "Circular":
        size = np.array([ settings['roi_diameter'],
                          settings['roi_diameter'],
                          settings['roi_height'] ], np.float32)
    elif machine_shape == "Rectangular":
        size = np.array([ settings['roi_width'],
                          settings['roi_depth'],
                          settings['roi_height'] ], np.float32)
    return get_size_polygons(size, machine_shape)


def get_size_polygons(size, machine_shape):
    ret = []
    if machine_shape == 'Circular':
        circle = []
        steps = 32
        for n in range(0, steps):
            circle.append([math.cos(float(n) / steps * 2 * math.pi) * size[0] / 2,
                           math.sin(float(n) / steps * 2 * math.pi) * size[1] / 2])
        ret.append(np.array(circle, np.float32))

    elif machine_shape == 'Rectangular':
        rectangle = [[-size[0] / 2, size[1] / 2], [size[0] / 2, size[1] / 2], [size[0] / 2, -size[1] / 2],
                     [-size[0] / 2, -size[1] / 2]]
        ret.append(np.array(rectangle, np.float32))

    w = 20
    h = 20
    ret.append(np.array([[-size[0] / 2, -size[1] / 2],
                         [-size[0] / 2 + w + 2, -size[1] / 2],
                         [-size[0] / 2 + w, -size[1] / 2 + h],
                         [-size[0] / 2, -size[1] / 2 + h]], np.float32))
    ret.append(np.array([[size[0] / 2 - w - 2, -size[1] / 2],
                         [size[0] / 2, -size[1] / 2],
                         [size[0] / 2, -size[1] / 2 + h],
                         [size[0] / 2 - w, -size[1] / 2 + h]], np.float32))
    ret.append(np.array([[-size[0] / 2 + w + 2, size[1] / 2],
                         [-size[0] / 2, size[1] / 2],
                         [-size[0] / 2, size[1] / 2 - h],
                         [-size[0] / 2 + w, size[1] / 2 - h]], np.float32))
    ret.append(np.array([[size[0] / 2, size[1] / 2],
                         [size[0] / 2 - w - 2, size[1] / 2],
                         [size[0] / 2 - w, size[1] / 2 - h],
                         [size[0] / 2, size[1] / 2 - h]], np.float32))

    return ret

laser_bg_scanning = [None, None]
laser_bg_scanning_enable = False

laser_bg_calibration = [None, None]
laser_bg_calibration_enable = False

