# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

from horus.engine.driver.driver import Driver
from horus.engine.scan.ciclop_scan import CiclopScan
from horus.engine.scan.current_video import CurrentVideo
from horus.engine.calibration.pattern import Pattern
from horus.engine.calibration.calibration_data import CalibrationData
from horus.engine.calibration.camera_intrinsics import CameraIntrinsics
from horus.engine.calibration.autocheck import Autocheck
from horus.engine.calibration.laser_triangulation import LaserTriangulation
from horus.engine.calibration.platform_extrinsics import PlatformExtrinsics
from horus.engine.calibration.combo_calibration import ComboCalibration
from horus.engine.algorithms.image_capture import ImageCapture
from horus.engine.algorithms.image_detection import ImageDetection
from horus.engine.algorithms.laser_segmentation import LaserSegmentation
from horus.engine.algorithms.point_cloud_generation import PointCloudGeneration
from horus.engine.algorithms.point_cloud_roi import PointCloudROI


# Instances of engine modules

driver = Driver()
ciclop_scan = CiclopScan()
current_video = CurrentVideo() # no params
pattern = Pattern()
calibration_data = CalibrationData()
camera_intrinsics = CameraIntrinsics() # no params
scanner_autocheck = Autocheck() # no params
laser_triangulation = LaserTriangulation()
platform_extrinsics = PlatformExtrinsics()
combo_calibration = ComboCalibration()
image_capture = ImageCapture()
image_detection = ImageDetection() # no params
laser_segmentation = LaserSegmentation()
point_cloud_generation = PointCloudGeneration() # no params
point_cloud_roi = PointCloudROI()

"""
engine_mode = ''

def setup_engine_params(mode = 'calibration'):
    global engine_mode
    engine_mode = mode

    driver.camera.read_profile()

    driver.board.motor_speed(profile.settings['motor_speed_control'])
    driver.board.motor_acceleration(profile.settings['motor_acceleration_control'])

    # =========== image_capture ============
    

    if mode == 'control':
        image_capture.texture_mode.read_profile('control')

    elif mode == 'calibration':
        image_capture.texture_mode.read_profile('texture_scanning')
        image_capture.pattern_mode.read_profile('pattern_calibration') # pattern_mode have only 'calibration' profile
        image_capture.laser_mode.read_profile('laser_calibration')
        image_capture.set_remove_background(profile.settings['remove_background_calibration'])

    elif mode == 'scanning':
        image_capture.texture_mode.read_profile('texture_scanning')
        image_capture.pattern_mode.read_profile('pattern_calibration') # pattern_mode have only 'calibration' profile
        image_capture.laser_mode.read_profile('laser_scanning')
        image_capture.set_remove_background(profile.settings['remove_background_scanning'])

    else:
        pass

    image_capture.set_use_distortion(profile.settings['use_distortion'])

    # ============== laser_segmentation ==============
    laser_segmentation.read_profile(mode)

    # =============== pattern =============
    pattern.read_profile()

    # ============= calibration_data ==========
    width, height = driver.camera.get_resolution()
    calibration_data.set_resolution(width, height)
    calibration_data.read_profile()

    # ============== laser_triangulation ==============
    laser_triangulation.read_profile()

    # ============== platform_extrinsics ==============
    platform_extrinsics.read_profile()

    # =========== combo_calibration ==============
    combo_calibration.read_profile()

    # =============== ciclop_scan ================
    ciclop_scan.read_profile()

    # ============= point_cloud_roi ================
    point_cloud_roi.read_profile()



def switch_engine_params(mode='calibration'): # 'scanning'
    global engine_mode
    if engine_mode == mode:
        return

    engine_mode = mode

    laser_mode = image_capture.laser_mode
    laser_mode.brightness = profile.settings['brightness_laser_'+mode]
    laser_mode.contrast = profile.settings['contrast_laser_'+mode]
    laser_mode.saturation = profile.settings['saturation_laser_'+mode]
    laser_mode.exposure = profile.settings['exposure_laser_'+mode]
    laser_mode.set_light(1,profile.settings['light1_laser_'+mode])
    laser_mode.set_light(2,profile.settings['light2_laser_'+mode])

    image_capture.set_remove_background(profile.settings['remove_background_'+mode])

    laser_segmentation.red_channel = profile.settings['red_channel_'+mode]
    laser_segmentation.threshold_enable = profile.settings['threshold_enable_'+mode]
    laser_segmentation.threshold_value = profile.settings['threshold_value_'+mode]
    laser_segmentation.blur_enable = profile.settings['blur_enable_'+mode]
    laser_segmentation.set_blur_value(profile.settings['blur_value_'+mode])
    laser_segmentation.window_enable = profile.settings['window_enable_'+mode]
    laser_segmentation.window_value = profile.settings['window_value_'+mode]
    laser_segmentation.refinement_method = profile.settings['refinement_'+mode]
"""



