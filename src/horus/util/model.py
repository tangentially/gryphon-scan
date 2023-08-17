# -*- coding: utf-8 -*-
# This file is part of the Horus Project
from numpy.core._multiarray_umath import ndarray

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.\
                 Copyright (C) 2013 David Braam from Cura Project'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import os

import numpy as np
np.seterr(all='ignore')


class Model(object):
    """
    Each object has a Mesh and a 3x3 transformation matrix to rotate/scale the object.
    """

    def __init__(self, origin_filename, is_point_cloud=False):
        self._origin_filename = origin_filename
        self._is_point_cloud = is_point_cloud

        if origin_filename is None:
            self._name = 'None'
        else:
            self._name = os.path.basename(origin_filename)
        if '.' in self._name:
            self._name = os.path.splitext(self._name)[0]
        self._mesh = None
        self._position = np.array([0.0, 0.0, 0.0])
        self._matrix = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
        self._min = None
        self._max = None
        self._size = np.array([0.0, 0.0, 0.0])
        self._boundary_circle_size = 75.0
        self._draw_offset = np.array([0.0, 0.0, 0.0])

    def _add_mesh(self):
        self._mesh = Mesh(self)
        return self._mesh

    def _post_process_after_load(self):
        if len(self._mesh.vertexes) > 0:
            if not self._is_point_cloud:
                self._mesh._calculate_normals()

            self._min = np.array([np.inf, np.inf, np.inf], np.float64)
            self._max = np.array([-np.inf, -np.inf, -np.inf], np.float64)
            self._boundary_circle_size = 0

            vertexes = self._mesh.vertexes
            vmin = vertexes.min(0)
            vmax = vertexes.max(0)
            for n in range(0, 3):
                self._min[n] = min(vmin[n], self._min[n])
                self._max[n] = max(vmax[n], self._max[n])

            # Calculate the boundary circle
            center = vmin + (vmax - vmin) / 2.0
            boundary_circle_size = round(np.max(np.linalg.norm(vertexes - center, axis=1)), 3)
            self._boundary_circle_size = max(self._boundary_circle_size, boundary_circle_size)

            self._size = self._max - self._min
            if not self._is_point_cloud:
                self._draw_offset = (self._max + self._min) / 2
                self._draw_offset[2] = self._min[2]
            self._max -= self._draw_offset
            self._min -= self._draw_offset

    def get_position(self):
        return self._position

    def get_matrix(self):
        return self._matrix

    def get_size(self):
        return self._size

    def get_draw_offset(self):
        return self._draw_offset

    def get_boundary_circle(self):
        return self._boundary_circle_size

    def is_point_cloud(self):
        return self._is_point_cloud

    def get_scale(self):
        return np.array([
            np.linalg.norm(self._matrix[::, 0].getA().flatten()),
            np.linalg.norm(self._matrix[::, 1].getA().flatten()),
            np.linalg.norm(self._matrix[::, 2].getA().flatten())], np.float64)


class Mesh(object):
    """
    A mesh is a list of 3D triangles build from vertexes.
    Each triangle has 3 vertexes. It can be also a point cloud.
    A "VBO" can be associated with this object, which is used for rendering this object.
    """
    vertexes_meta = None  # type: ndarray

    def __init__(self, obj = None):
        self.vertexes = np.zeros((0, 3), np.float32)
        self.vertexes_meta = np.empty((0,), dtype=[('laser_id',np.int8),('slice_no','int'),('slice_l',np.float32)])
        self.colors = np.zeros((0, 3), np.uint8)
        self.normal = np.zeros((0, 3), np.float32)
        self.vertex_count = 0

        self.vbo = None
        self._obj = obj
        self.current_cloud_index = 0
        self.metadata = None

    def _add_vertex(self, x, y, z, r=255, g=255, b=255, laser_index=None, slice_no = None, slice_l = None):
        if laser_index is None:
            laser_index=self.current_cloud_index
        n = self.vertex_count
        # TODO extend array if required
        self.vertexes[n] = (x, y, z)
        self.colors[n]   = (r, g, b)
        self.vertexes_meta[n] = (laser_index, slice_no, slice_l)
        self.vertex_count += 1

    def add_pointcloud(self, cloud_vertex, cloud_color, meta=None ):
        if cloud_vertex is None or \
           cloud_vertex.shape[0] <= 0:
            return
        #print "Add {0} to {1}".format(cloud_vertex.shape, self.vertexes.shape)
        #if laser_index < 0:
        #    laser_index=self.current_cloud_index

        _meta = np.empty((), dtype=object)
        if meta is None:
            #_meta[()] = (laser_index, slice_no, slice_l)
            _meta[()] = (-1, -1, np.nan)
        else:
            _meta[()] = meta
        _meta = np.full(cloud_vertex.shape[0], _meta)

        n = self.vertex_count
        m = n + cloud_vertex.shape[0]
        if m >= self.vertexes.shape[0]:
            if self.vertexes.shape[0] > n:
                # shrink and append
                self.vertexes.resize((n,3))
                self.colors.resize((n,3))
                #self.normal.resize((n,3))
                self.vertexes_meta.resize(n, refcheck=False) #+self.vertexes_meta.shape[1:])

            self.vertexes      = np.append( self.vertexes,      cloud_vertex, axis=0)
            self.colors        = np.append( self.colors,        cloud_color,  axis=0)
            self.vertexes_meta = np.append( self.vertexes_meta, _meta,        axis=0)
        else:
            self.vertexes[n:m] = cloud_vertex
            self.colors[n:m] = cloud_color
            self.vertexes_meta[n:m] = _meta

        self.vertex_count = m

    def _add_face(self, x0, y0, z0, x1, y1, z1, x2, y2, z2):
        n = self.vertex_count
        self.vertexes[n], self.vertexes[
            n + 1], self.vertexes[n + 2] = (x0, y0, z0), (x1, y1, z1), (x2, y2, z2)
        self.vertex_count += 3

    def _prepare_vertex_count(self, vertex_number):
        # Set the amount of vertex before loading data in them. This way we can
        # create the np arrays before we fill them.
        self.vertexes = np.zeros((vertex_number, 3), np.float32)
        self.colors = np.zeros((vertex_number, 3), np.int32)
        self.normal = np.zeros((vertex_number, 3), np.float32)
        meta = np.empty((), dtype=object)
        meta[()] = (-1, -1, np.nan)
        self.vertexes_meta = np.full(vertex_number, meta, dtype=self.vertexes_meta.dtype)
        self.vertex_count = 0
        return self

    def _prepare_face_count(self, face_number):
        # Set the amount of faces before loading data in them. This way we can
        # create the np arrays before we fill them.
        self.vertexes = np.zeros((face_number * 3, 3), np.float32)
        self.normal = np.zeros((face_number * 3, 3), np.float32)
        self.vertex_count = 0

    def _calculate_normals(self):
        # Calculate the normals
        tris = self.vertexes.reshape((self.vertex_count / 3, 3, 3))
        normals = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
        normals /= np.linalg.norm(normals)
        n = np.concatenate((np.concatenate((normals, normals), axis=1), normals), axis=1)
        self.normal = n.reshape(self.vertex_count, 3)

    def get_vertexes(self):
        return self.vertexes[0:self.vertex_count]

    def get_meta(self):
        return self.vertexes_meta[0:self.vertex_count]

    def copy(self, mesh):
        self.vertexes      = np.copy(mesh.vertexes)
        self.vertexes_meta = np.copy(mesh.vertexes_meta)
        self.colors        = np.copy(mesh.colors)
        self.normal        = np.copy(mesh.normal)
        self.vertex_count  = mesh.vertex_count

        self.vbo = None
        self._obj = mesh._obj
        self.current_cloud_index = mesh.current_cloud_index
        if mesh.metadata is not None:
            self.metadata = mesh.metadata.copy()

        return self

    def clear_vbo(self):
        if self.vbo is not None:
            self.vbo.release()
            self.vbo = None


