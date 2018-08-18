# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Mikhail N Klimushin aka Night Gryphon <ngryph@gmail.com>'
__copyright__ = 'Copyright (C) 2018 Night Gryphon'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np
import math
from horus.util import profile
#from horus.gui.engine import platform_extrinsics, image_detection, pattern
import horus.gui.engine

platform_points = None
platform_border = None

# Platform visualization cache shapes
def init_platform_augmented_draw():
    global platform_points
    global platform_border

    r = profile.settings['platform_markers_diameter']/2 #138
    h = profile.settings['platform_markers_z']
    rr = r * np.sin(np.deg2rad(45))
    platform_points = np.float32( [(0,0,0),(r,0,h),(-r,0,h),(0,r,h),(0,-r,h),(rr,rr,h),(rr,-rr,h),(-rr,rr,h),(-rr,-rr,h)])

    h = profile.settings['machine_shape_z']
    polys = profile.get_machine_size_polygons(profile.settings["machine_shape"])
    platform_border = []
    for p in polys[0]:
        platform_border.append( ( p[0], p[1], h) )
    platform_border = np.float32(platform_border)

# Platform visualization draw
def augmented_draw_platform(image):
    global platform_points
    global platform_border

    if platform_points is None:
        init_platform_augmented_draw()

    if horus.gui.engine.platform_extrinsics.calibration_data.platform_rotation is not None and \
       horus.gui.engine.platform_extrinsics.calibration_data.platform_translation is not None:
        #cv2.circle(image, (50,50), 5, (0,0,255), -1)
        # platform border
        p, jac = cv2.projectPoints(platform_border, \
            horus.gui.engine.platform_extrinsics.calibration_data.platform_rotation, \
            horus.gui.engine.platform_extrinsics.calibration_data.platform_translation, \
            horus.gui.engine.platform_extrinsics.calibration_data.camera_matrix, \
            horus.gui.engine.platform_extrinsics.calibration_data.distortion_vector)
        p = np.int32([p])
        cv2.polylines(image, p, True, (0,255,0), 2)

        # marker positions
        p, jac = cv2.projectPoints(platform_points, \
            horus.gui.engine.platform_extrinsics.calibration_data.platform_rotation, \
            horus.gui.engine.platform_extrinsics.calibration_data.platform_translation, \
            horus.gui.engine.platform_extrinsics.calibration_data.camera_matrix, \
            horus.gui.engine.platform_extrinsics.calibration_data.distortion_vector)
        p = np.int32(p).reshape(-1,2)
        for pp in p:
            cv2.circle(image, tuple(pp), 5, (0,0,255), -1)

# Platform remove mask
def augmented_platform_mask(image):
    global platform_points
    global platform_border

    mask = np.ones(image.shape[0:2], dtype = "uint8")
    if platform_points is None:
        init_platform_augmented_draw()

    if horus.gui.engine.platform_extrinsics.calibration_data.platform_rotation is not None and \
       horus.gui.engine.platform_extrinsics.calibration_data.platform_translation is not None:
        # platform border
        p, jac = cv2.projectPoints(platform_border, \
            horus.gui.engine.platform_extrinsics.calibration_data.platform_rotation, \
            horus.gui.engine.platform_extrinsics.calibration_data.platform_translation, \
            horus.gui.engine.platform_extrinsics.calibration_data.camera_matrix, \
            horus.gui.engine.platform_extrinsics.calibration_data.distortion_vector)
        p = np.int32([p])
        cv2.fillPoly(mask, p, 0)
    return mask

#pattern visualization
def augmented_draw_pattern(image, corners):
    if corners is not None:
        cv2.drawChessboardCorners(
            image, (horus.gui.engine.pattern.columns, horus.gui.engine.pattern.rows), corners, True)

        pose = horus.gui.engine.image_detection.detect_pose_from_corners(corners)
        l = -horus.gui.engine.pattern.square_width
        t = -horus.gui.engine.pattern.square_width
        r = horus.gui.engine.pattern.square_width * horus.gui.engine.pattern.columns
        b = horus.gui.engine.pattern.square_width * horus.gui.engine.pattern.rows
        wl = horus.gui.engine.pattern.border_l
        wr = horus.gui.engine.pattern.border_r
        wt = horus.gui.engine.pattern.border_t
        wb = horus.gui.engine.pattern.border_b
        points = np.float32( (
            (l,t,0),(r,t,0),(r,b,0),(l,b,0),
            (l-wl,t-wt,0),(r+wr,t-wt,0),(r+wr,b+wb,0),(l-wl,b+wb,0),
            (l-wl,b-horus.gui.engine.pattern.square_width+horus.gui.engine.pattern.origin_distance,0),(r+wr,b-horus.gui.engine.pattern.square_width+horus.gui.engine.pattern.origin_distance,0)
            ) )
        p, jac = cv2.projectPoints(points, \
            pose[0], \
            pose[1].T[0], \
            horus.gui.engine.platform_extrinsics.calibration_data.camera_matrix, \
            horus.gui.engine.platform_extrinsics.calibration_data.distortion_vector)
        p = np.int32(p).reshape(-1,2)
        cv2.polylines(image, np.int32([p[0:4]]), True, (0,255,0), 2)
        cv2.polylines(image, np.int32([p[4:8]]), True, (255,0,0), 2)
        cv2.line(image, tuple(p[8]), tuple(p[9]), (255,0,0), 2)
    return image

#pattern mask
def augmented_pattern_mask(image, corners):
    mask = np.zeros(image.shape[0:2], dtype = "uint8")
    if corners is not None:
        pose = horus.gui.engine.image_detection.detect_pose_from_corners(corners)
        l = -horus.gui.engine.pattern.square_width
        t = -horus.gui.engine.pattern.square_width
        r = horus.gui.engine.pattern.square_width * horus.gui.engine.pattern.columns
        b = horus.gui.engine.pattern.square_width * horus.gui.engine.pattern.rows
        wl = horus.gui.engine.pattern.border_l
        wr = horus.gui.engine.pattern.border_r
        wt = horus.gui.engine.pattern.border_t
        wb = horus.gui.engine.pattern.border_b
        points = np.float32( (
            (l-wl,t-wt,0),(r+wr,t-wt,0),(r+wr,b+wb,0),(l-wl,b+wb,0),
            ) )
        p, jac = cv2.projectPoints(points, \
            pose[0], \
            pose[1].T[0], \
            horus.gui.engine.platform_extrinsics.calibration_data.camera_matrix, \
            horus.gui.engine.platform_extrinsics.calibration_data.distortion_vector)
        p = np.int32(p).reshape(-1,2)

        cv2.fillConvexPoly(mask, np.int32([p]), 255)
    return mask


# Apply mask to image
def apply_mask(image, mask):
    if image is not None and \
       mask  is not None:
        image = cv2.bitwise_and(image, image, mask=mask)
        return image


# Draw mask as color overlay
def overlay_mask(image, mask, color=(255,0,0)):
    if image is not None and \
       mask  is not None:
        overlayImg = np.zeros(image.shape, image.dtype)
        overlayImg[:,:] = color
        overlayMask = cv2.bitwise_and(overlayImg, overlayImg, mask=mask)
        cv2.addWeighted(overlayMask, 1, image, 1, 0, image)


# estimate platform rotate angle to to make pattern on platform perpendicular to camera
def estimate_platform_angle_from_pattern(pose):
#	pose = horus.gui.engine.image_detection.detect_pose_from_corners(corners)
    if pose is not None:
        if horus.gui.engine.platform_extrinsics.calibration_data.platform_rotation is not None and \
           horus.gui.engine.platform_extrinsics.calibration_data.platform_translation is not None:
            # Platform position known. Calculate angle in platform space

            # pattern normal in camera space
            pc = np.float32([(0,0,-1)]).dot(pose[0].T) 
            # add camera-to-platform vector
            vv = np.append(pc, [-horus.gui.engine.platform_extrinsics.calibration_data.platform_translation / np.linalg.norm(horus.gui.engine.platform_extrinsics.calibration_data.platform_translation)], axis=0)
            # move all to platform space
            pz = vv.dot(horus.gui.engine.platform_extrinsics.calibration_data.platform_rotation)
            # flattern to platform XY / normalize
            pz[:,2] = 0
            pz /= np.apply_along_axis(np.linalg.norm, 1, pz)[..., np.newaxis]
            return math.copysign(np.rad2deg(math.acos(np.dot(pz[0], pz[1]))) , pz[0,1]-pz[1,1])

        else:
            # platform position unknown. trying to do the best
            # expect pattern Y is close to platform normal and platform-to-camera about perpendicular to camera

            # camera Z to pattern space
            v = np.float32([(0,0,1)]).dot(pose[0]) # camera Z axis to pattern space
            return math.copysign(np.rad2deg(math.acos(v[0,2]/np.linalg.norm( (v[0,0], v[0,2]) ))), v[0,0]) 
