# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np
from itertools import cycle

from horus import Singleton


@Singleton
class CurrentVideo(object):

    def __init__(self):
        self.mode = 'Texture'

        self.images = {}
        self.images['Texture'] = None
        self.images['Laser'] = None
        self.images['Gray'] = None
        self.images['Line'] = None

    def set_texture(self, image):
        self.images['Texture'] = image

    def set_laser(self, images):
        image = self._combine_images(images)
        self.images['Laser'] = image

    def set_gray(self, images):
        if images is not None:
            image = self._combine_images(images)
            if image is not None:
                image = cv2.merge((image, image, image))
        self.images['Gray'] = image

    def set_line(self, points, image):
        if image is None:
            return

        line_colors = cycle([[255,0,0],[0,255,255],[0,255,0],[255,0,255]])
        lines = np.zeros_like(image)
        for p in points:
            c = next(line_colors)
            if p:
                lines[p[1].astype(int), np.around(p[0]).astype(int)] = c

        self.images['Line'] = cv2.addWeighted(image,0.5,lines,1.,0.)

    def _combine_images(self, images):
        im = [i for i in images if i is not None]
        if len(im)>0:
            return np.max(im, axis=0)

        return None

    def _compute_line_image(self, points, image):
        if image is None:
            return None
        if points is not None:
            u, v = points
            image = np.zeros_like(image)
            #image[v.astype(int), np.around(u).astype(int) - 1] = 255
            image[v.astype(int), np.around(u).astype(int)] = 255
            #image[v.astype(int), np.around(u).astype(int) + 1] = 255
            return image

    def capture(self):
        return self.images[self.mode]
