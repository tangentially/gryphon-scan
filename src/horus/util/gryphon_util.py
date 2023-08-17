# -*- coding: utf-8 -*-
# This file is part of the Gryphon Scan Project

__author__ = 'Mikhail N Klimushin aka Night Gryphon <ngryph@gmail.com>'
__copyright__ = 'Copyright (C) 2018 Night Gryphon'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
from itertools import chain
import math
import numpy as np
import struct
from scipy import optimize

import horus.gui.engine

# ================================================
# Apply mask to image

def apply_mask(image, mask):
    if image is not None and \
       mask  is not None:
        image = cv2.bitwise_and(image, image, mask=mask)
        return image


# ================================================
# Draw mask as color overlay

def overlay_mask(image, mask, color=(255,0,0)):
    if image is not None and \
       mask  is not None:
        overlayImg = np.zeros(image.shape, image.dtype)
        overlayImg[:,:] = color
        overlayMask = cv2.bitwise_and(overlayImg, overlayImg, mask=mask)
        cv2.addWeighted(overlayMask, 1, image, 1, 0, image)


# ================================================
# Multiple measurements corner capture

def capture_precise_corners(steps = 3):
    corners = []
    image_capture = horus.gui.engine.image_capture
    stream_save = image_capture.stream
    image_capture.stream = False
    for i in range(steps):
        print(i)
        image = image_capture.capture_pattern()
        if image is not None:
            c = horus.gui.engine.image_detection.detect_corners(image)
            if c is not None:
                corners += [c] # [c.reshape(-1,2)]
            elif i>=1 and len(corners) == 0:
                # two captures with no corners
                return image, None, None
    image_capture.stream = stream_save
    if len(corners) == 0:
        return image, None, None
    corners = np.float32(corners)
    return image, np.mean(corners, axis=0), np.max(np.std(corners, axis=0))
        

# ================================================
# check and decode color setting to RGB 

def decode_color(value, default=(0,0,0)):
    ret = default
    if isinstance(value, str):
        ret = struct.unpack('BBB', bytes.fromhex(value))
    elif isinstance(value, (tuple,list)) and \
         len(value) == 3 and \
         all(isinstance(x, int) for x in value):
       ret = value

    return ret


# ================================================
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


# ================================================
# Convert plane from Rmat+t to n+d format
# m - rotation matrix, t - translation

def pos2nd(m,t):
    if m is None or t is None:
        return (None, None)
                
    norm = m.T[2] 
    norm[:] /= np.linalg.norm(norm)
    dist = t.dot(norm.T)

    return (norm, dist)


# ================================================
# compute plane cross line

def plane_cross(n1, d1, n2, d2):
    if n1 is None or d1 is None or n2 is None or d2 is None:
        return (None, None)
    vec = np.cross(n1, n2)
    vec /= np.linalg.norm(vec)
#    A = np.array([n1, n2, vec])
    A = np.array([n1, n2, (0,0,1)])
    d = np.array([d1, d2, 0 ])
    pt = np.linalg.solve(A,d).T

    return (vec ,pt)


# ================================================
# compute line and sphere cross points
# vec, pt - line
# center, radius - sphere

def line_cross_sphere(vec, pt, center, radius):
    pc = center - pt
    pc_len = np.linalg.norm(pc)
    vec /= np.linalg.norm(vec)
    pn = vec.dot(pc.T)
    d2 = radius*radius - (pc_len*pc_len - pn*pn)
    if d2 < 0:
       return None, None
            
    delta = np.sqrt(d2)
    p1 = pt + (pn+delta)*vec
    p2 = pt + (pn-delta)*vec
    return p1, p2


# ================================================
# find platform rotation angle to move point A to plane n,d
# A - point of platform object in world coords
# n,d - plane distance and normal in world coords
# return - platform angle movement to rotate point on to plane

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

    
# ================================================
# Finding optimal rotation and translation between corresponding 3D points
# http://nghiaho.com/?page_id=671
#
# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    N = A.shape[0]  # total points
    print("Arr len: "+str(N))
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    print(centroid_A)
    print(centroid_B)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.matmul(np.transpose(AA), BB)

    U, S, Vt = np.linalg.svd(H)

    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       print("Reflection detected")
       Vt[2,:] *= -1
       R = np.matmul(Vt.T, U.T)

    t = -np.matmul(R,centroid_A.T) + centroid_B.T

    return R, t, centroid_A, centroid_B


# ================================================
# Point to line projection
#  a, v - line point and vector 
#  p - point to project

def PointOntoLine(a, v, p):
    ap = p-a
    result = a + np.dot(ap,v)/np.dot(v,v) * v
    return result


# ================================================
# find best fit line for 3D point set
# https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
# points - [X,Y,Z]

def FitLine3D(points):
    # Generate some data that lies along a line
    '''
    x = np.mgrid[-2:5:120j]
    y = np.mgrid[1:9:120j]
    z = np.mgrid[-5:3:120j]
    
    data = np.concatenate((x[:, np.newaxis], 
                           y[:, np.newaxis], 
                           z[:, np.newaxis]), 
                          axis=1)
    '''
    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = points.mean(axis=0)
    
    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(points - datamean)
    
    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.
    
    # shift by the mean to get the line in the right place
    return datamean, vv[0]


# -----------------------------------------------------------
# https://stackoverflow.com/questions/17973507/why-is-converting-a-long-2d-list-to-numpy-array-so-slow
def longlist2ndarray(longlist):
    flat = np.fromiter(chain.from_iterable(longlist), np.array(longlist[0][0]).dtype, -1) # Without intermediate list:)
    return flat.reshape((len(longlist), -1))


# -----------------------------------------------------------
def fit_plane_svd(points):
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    c = np.mean(points, axis=0)
    data = points.T-c
    u, sigma, v = np.linalg.svd(data)
    normal = v[2]                                 
    normal /= np.linalg.norm(normal)
    d = np.dot(normal, c)

    return c, normal, d


# ------------- fit plane with point = center of mass -------
def fit_plane_leastsq(points):
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    c = np.mean(points, axis=0)
    normal = fit_normal_leastsq(points - c)
    d = np.dot(normal, c)

    return c, normal, d


# --------- estimate plane normal for set of inplane vectors ----------
def residuals_normal(parameters, data_vectors):
    # minimize projections to estimating axis
    theta, phi = parameters
    v = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    distances =  [np.abs(np.dot(v, [x, y, z])) for x, y, z in data_vectors]
    return distances


def fit_normal_leastsq(data):
    estimate = [-np.pi/2, np.pi/2]  # theta, phi
    best_fit_values = optimize.leastsq(residuals_normal, estimate, args=(data*1000))
    tF, pF = best_fit_values
    v = np.array([np.sin(tF) * np.cos(pF), np.sin(tF) * np.sin(pF), np.cos(tF)])
    return v

