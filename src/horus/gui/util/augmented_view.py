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

    h = profile.settings['platform_border_z']
    polys = profile.get_machine_size_polygons()
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

        calibration_data = horus.gui.engine.platform_extrinsics.calibration_data
        # platform border
        p, jac = cv2.projectPoints(platform_border, \
            calibration_data.platform_rotation, \
            calibration_data.platform_translation, \
            calibration_data.camera_matrix, \
            calibration_data.distortion_vector)
        p = np.int32([p])
        cv2.polylines(image, p, True, (0,255,0), 2)

        # marker positions
        p, jac = cv2.projectPoints(platform_points, \
            calibration_data.platform_rotation, \
            calibration_data.platform_translation, \
            calibration_data.camera_matrix, \
            calibration_data.distortion_vector)
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

        calibration_data = horus.gui.engine.platform_extrinsics.calibration_data
        # platform border
        p, jac = cv2.projectPoints(platform_border, \
            calibration_data.platform_rotation, \
            calibration_data.platform_translation, \
            calibration_data.camera_matrix, \
            calibration_data.distortion_vector)
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

        calibration_data = horus.gui.engine.platform_extrinsics.calibration_data
        points = np.float32( (
            (l,t,0),(r,t,0),(r,b,0),(l,b,0),
            (l-wl,t-wt,0),(r+wr,t-wt,0),(r+wr,b+wb,0),(l-wl,b+wb,0),
            (l-wl,b-horus.gui.engine.pattern.square_width+horus.gui.engine.pattern.origin_distance,0),(r+wr,b-horus.gui.engine.pattern.square_width+horus.gui.engine.pattern.origin_distance,0),
            (l,b,0),(l,b,-50)
            ) )
        p, jac = cv2.projectPoints(points, \
            pose[0], \
            pose[1].T[0], \
            calibration_data.camera_matrix, \
            calibration_data.distortion_vector)
        p = np.int32(p).reshape(-1,2)
        cv2.polylines(image, np.int32([p[0:4]]), True, (0,255,0), 2)
        cv2.polylines(image, np.int32([p[4:8]]), True, (255,0,0), 2)
        cv2.line(image, tuple(p[8]), tuple(p[9]), (255,0,0), 2)
        cv2.line(image, tuple(p[10]), tuple(p[11]), (255,0,0), 2)

        cv2.putText(image, str(pose[1].T[0]), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), lineType=cv2.LINE_AA)
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

        calibration_data = horus.gui.engine.platform_extrinsics.calibration_data
        points = np.float32( (
            (l-wl,t-wt,0),(r+wr,t-wt,0),(r+wr,b+wb,0),(l-wl,b+wb,0),
            ) )
        p, jac = cv2.projectPoints(points, \
            pose[0], \
            pose[1].T[0], \
            calibration_data.camera_matrix, \
            calibration_data.distortion_vector)
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
def estimate_platform_angle_from_pattern(pose, use_camera_space = False):
#	pose = horus.gui.engine.image_detection.detect_pose_from_corners(corners)
    calibration = horus.gui.engine.platform_extrinsics.calibration_data
    if pose is not None:
        if calibration.platform_rotation is not None and \
           np.count_nonzero(calibration.platform_rotation) > 0 and \
           calibration.platform_translation is not None and \
           np.count_nonzero(calibration.platform_translation) > 0 and \
           not use_camera_space:
            # Platform position known. Calculate angle in platform space

            # pattern normal in camera space
            pc = np.float32([(0,0,-1)]).dot(pose[0].T)
            # add camera-to-platform vector
            vv = np.append(pc, [-calibration.platform_translation / np.linalg.norm(calibration.platform_translation)], axis=0)
            # move all to platform space
            pz = vv.dot(calibration.platform_rotation)
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
    return None

# draw laser plane projection
def augmented_draw_lasers_on_platform(image):
    if image is None:
        return
    calibration = horus.gui.engine.platform_extrinsics.calibration_data
    if calibration.platform_rotation is not None and \
       calibration.platform_translation is not None:
            p_norm, p_dist = pos2nd(calibration.platform_rotation, calibration.platform_translation)
            for laser in calibration.laser_planes:
                if not laser.is_empty():
                    line_vec, line_p = plane_cross(p_norm, p_dist, laser.normal, laser.distance)
                    
                    p1, p2 = line_cross_sphere(line_vec, line_p, 
                        calibration.platform_translation, 
                        profile.settings['machine_diameter']/2 )
                    points = np.float32([p1, p2])
                    p, jac = cv2.projectPoints(points,
                        np.identity(3),
                        np.zeros(3),
                        calibration.camera_matrix,
                        calibration.distortion_vector)
                    p = np.int32(p).reshape(-1,2)
                    cv2.line(image, tuple(p[0]), tuple(p[1]), (255,0,0), 2)


def augmented_draw_lasers_on_pattern(image, pose):
    if image is None or pose is None:
        return

    calibration = horus.gui.engine.platform_extrinsics.calibration_data
    if calibration.laser_planes is None:
        return

    p = horus.gui.engine.pattern
#    pl = -p.square_width - p.border_l
    pt = -p.square_width - p.border_t
#    pr = p.square_width * p.columns + p.border_r
    pb = p.square_width * p.rows + p.border_b

    r_inv = np.linalg.inv(pose[0])

    for laser in calibration.laser_planes:
        if laser.normal is None or laser.distance is None:
            continue

        l_n = r_inv.dot(laser.normal)
        pp = laser.distance*laser.normal[:]-pose[1].T[0]
        l_d = pp.dot(laser.normal.T)

        if l_n[0] != 0:
            xt = (l_d-pt*l_n[1])/l_n[0]
            xb = (l_d-pb*l_n[1])/l_n[0]
            points = np.float32([ (xt,pt,0),(xb,pb,0) ])
            p, jac = cv2.projectPoints(points,
                pose[0],
                pose[1].T[0],
                calibration.camera_matrix,
                calibration.distortion_vector)
            p = np.int32(p).reshape(-1,2)
            cv2.line(image, tuple(p[0]), tuple(p[1]), (255,0,0), 2)


def pos2nd(m,t):
    # m - rotation matrix, t - translation
    if m is None or t is None:
        return (None, None)
                
    norm = m.T[2] 
    norm[:] /= np.linalg.norm(norm)
    dist = t.dot(norm.T)

    return (norm, dist)


def plane_cross(n1, d1, n2, d2):
    if n1 is None or d1 is None or n2 is None or d2 is None:
        return (None, None)

    vec = np.cross(n1, n2)
#    A = np.array([n1, n2, vec])
    A = np.array([n1, n2, (0,0,1)])
    d = np.array([d1, d2, 0 ])
    pt = np.linalg.solve(A,d).T

    return (vec ,pt)


def line_cross_sphere(vec, pt, center, radius):
    pc = center - pt
    pc_len = np.linalg.norm(pc)
    pn = vec.dot(pc.T)
    d2 = radius*radius - (pc_len*pc_len - pn*pn)
    if d2 < 0:
       return None, None
            
    delta = np.sqrt(d2)
    p1 = pt + (pn+delta)*vec
    p2 = pt + (pn-delta)*vec
    return p1, p2


# find platform rotation angle to move point A to plane n,d
# A - point of platform object in world coords
# n,d - plane distance and normal in world coords
# return - planform angle movement to rotate point on to plane
def rotatePoint2Plane(A,n,d):
    calibration_data = horus.gui.engine.platform_extrinsics.calibration_data
    P = calibration_data.platform_translation # platform center
    M = calibration_data.platform_rotation    # platform to world matrix

    A3 = M.T.dot(A - P) # A in platform coords
    n3 = M.T.dot(n) # plane in platform coords
    d3 = d - np.dot(P, n)
    # AA dot n3 = d3      - AA belongs to plane
    # AA dot z = A dot z  - rotate around Z axis
    # |AA| = |A|          - keep length
    # cos l = AA[0,1] dot A[0,1]
    # --------------
    # K = (d - Az*nz) / ny
    # Y = K - X * nx/ny
    K = (d3 - A3[2]*n3[2])/n3[1]

    # (1 + nx^2/ny^2) * X^2 - X * 2*K*nx/ny + K^2 - Ax^2 - Ay^2 = 0
    # J = K^2 - Ax^2 - Ay^2
    a = 1 + n3[0]**2/n3[1]**2
    b = -2*K*n3[0]/n3[1]
    c = K**2 - A3[0]**2 - A3[1]**2
    discr = b**2 - 4*a*c
    if discr >= 0:
        x1 = (-b + math.sqrt(discr)) / (2 * a)
        x2 = (-b - math.sqrt(discr)) / (2 * a)
        #print("x1 = %.2f \nx2 = %.2f" % (x1, x2))

        y1 = K - x1*n3[0]/n3[1]
        y2 = K - x2*n3[0]/n3[1]
    
        AA1 = [x1, y1]
        AA2 = [x2, y2]
        A4 = A3[0:2]
        #print('A1,2,4 '+str(np.round(AA1,3))+str(np.round(AA2,3))+str(np.round(A4,3))+" c1: "+str(np.cross(AA1,A4))+" c2: "+str(np.cross(AA1,A4)))

        #l1 = np.rad2deg(np.arccos( np.dot(AA1,A4) / np.linalg.norm(AA1) / np.linalg.norm(A4) ))
        #l2 = np.rad2deg(np.arccos( np.dot(AA2,A4) / np.linalg.norm(AA2) / np.linalg.norm(A4) ))
        #l1 = np.rad2deg(np.arcsin( np.cross(AA1,A4) / np.linalg.norm(AA1) / np.linalg.norm(A4) ))
        #l2 = np.rad2deg(np.arcsin( np.cross(AA2,A4) / np.linalg.norm(AA2) / np.linalg.norm(A4) ))

        if np.linalg.norm(A4/np.linalg.norm(A4) - AA1/np.linalg.norm(AA1)) < np.linalg.norm(A4/np.linalg.norm(A4) - AA2/np.linalg.norm(AA2)):
            return -np.rad2deg(np.arcsin( np.cross(AA1,A4) / np.linalg.norm(AA1) / np.linalg.norm(A4) ))
        else:
            return -np.rad2deg(np.arcsin( np.cross(AA2,A4) / np.linalg.norm(AA2) / np.linalg.norm(A4) ))

    return None
    
