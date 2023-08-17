# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

"""
PLY file point cloud loader.

    - Binary, which is easy and quick to read.
    - Ascii, which is harder to read, as can come with windows, mac and unix style newlines.

This module also contains a function to save objects as an PLY file.

http://en.wikipedia.org/wiki/PLY_(file_format)
"""

import struct
import numpy as np
import pickle

from horus import __version__
from horus.util import model

import logging
logger = logging.getLogger(__name__)


# -------------- Vertex -------------
def _load_ascii_vertex(mesh, stream, dtype, count):
    mesh._prepare_vertex_count(count)

    fields = dtype.names
    
    x = -1
    y = -1
    z = -1
    if 'x' in fields:
        x = fields.index('x')
    if 'y' in fields:
        y = fields.index('y')
    if 'z' in fields:
        z = fields.index('z')

    nx = -1
    ny = -1
    nz = -1
    if 'nx' in fields:
        nx = fields.index('nx')
    if 'ny' in fields:
        ny = fields.index('ny')
    if 'nz' in fields:
        nz = fields.index('nz')

    idx = -2
    if 'scalar_Original_cloud_index' in fields:
        idx = fields.index('scalar_Original_cloud_index')

    sn = -2
    sa = -1
    if 'slice_num' in fields:
        sn = fields.index('slice_num')
    if 'slice_angle' in fields:
        sa = fields.index('slice_angle')


    for i in range(count):
        data = stream.readline().split(' ')
        data.append(-1) # default value for '-2' index
        data.append(np.nan) # default value for '-1' index
        if data is not None:
            mesh._add_vertex(data[x], data[y], data[z], data[nx], data[ny], data[nz], data[idx], data[sn], data[sa])


def _load_binary_vertex(mesh, stream, dtype, count):
    data = np.fromfile(stream, dtype=dtype, count=count)

    fields = dtype.fields
    mesh.vertex_count = count

    if 'x' in fields:
        mesh.vertexes = np.array(zip(data['x'], data['y'], data['z']))
    else:
        mesh.vertexes = np.zeros((count, 3))

    if 'nx' in fields:
        mesh.normal = np.array(zip(data['nx'], data['ny'], data['nz']))
    else:
        mesh.normal = np.zeros((count, 3))

    if 'red' in fields:
        mesh.colors = np.array(zip(data['red'], data['green'], data['blue']))
    else:
        mesh.colors = 255 * np.ones((count, 3))

    if 'slice_index' in fields:
        slice_n = data['slice_index']
        slice_l = data['slice_angle']
        slice_n, slice_l = zip(*map(lambda x,y: (-1,np.nan) if x<0 else (x,y), slice_n, slice_l))
    else:
        slice_n = [-1]*count
        slice_l = [np.nan]*count

    if 'scalar_Original_cloud_index' in fields:
        cloud_index = data['scalar_Original_cloud_index']
    else:
        cloud_index = [-1]*count

    mesh.vertexes_meta = np.array(zip(cloud_index, slice_n, slice_l), dtype=mesh.vertexes_meta.dtype)


# ------------ Mesh Metadata ---------------
def _load_binary_metadata(mesh, stream, dtype, count):
    data = np.fromfile(stream, dtype=dtype, count=count)

    mesh.metadata = pickle.loads(data.view('S{0}'.format(count))[0])
    print(mesh.metadata)

def _load_ascii_metadata(mesh, stream, dtype, count):
    # This method didn't exist, I'm not sure what needs to be done here...
    # copying the above function code until debugging can happen
    data = np.fromfile(stream, dtype=dtype, count=count)

    mesh.metadata = pickle.loads(data.view('S{0}'.format(count))[0])
    print(mesh.metadata)


# ======================================
def _load_element(mesh, stream, format, element, dtype, count):
    print("Load elements: '{0}' x {1} format {2} @ {3}".format(element, count, format, stream.tell()))

    if len(dtype)<=0 or \
        element is None or \
        format is None or \
        count <= 0:
        return

    dtype = np.dtype(dtype)
    print("   Types: {0}".format(dtype.names))

    if format == 'ascii':
        if element == 'vertex':
            _load_ascii_vertex(mesh, stream, dtype, count)
        elif element == 'metadata':
            _load_ascii_metadata(mesh, stream, dtype, count)
        else:
            for i in range(count):
                stream.readline()

    elif format == 'binary_big_endian' or format == 'binary_little_endian':
        if element == 'vertex':
            _load_binary_vertex(mesh, stream, dtype, count)
        elif element == 'metadata':
            _load_binary_metadata(mesh, stream, dtype, count)
        else:
            np.fromfile(stream, dtype=dtype, count=count)

def load_scene(filename):
    obj = model.Model(filename, is_point_cloud=True)
    m = obj._add_mesh()
    with open(filename, "rb") as f:
        format = None
        line = None
        header = ''

        while line != 'end_header\n' and line != '':
            line = f.readline()
            header += line

        header = header.split('\n')

        if header[0] == 'ply':

            for line in header:
                if 'format ' in line:
                    format = line.split(' ')[1]
                    break

            if format is not None:
                if format == 'ascii':
                    fm = ''
                elif format == 'binary_big_endian':
                    fm = '>'
                elif format == 'binary_little_endian':
                    fm = '<'

            # PLY data types
            # https://web.archive.org/web/20161204152348/http://www.dcs.ed.ac.uk/teaching/cs4/www/graphics/Web/ply.html
            df = { 'float': fm + 'f4', \
                   'uchar': fm + 'B', \
                   'char': fm + 'b', \
                   'short': fm + 'i2', \
                   'ushort': fm + 'u2', \
                   'int': fm + 'i4', \
                   'uint': fm + 'u4', \
                   'double': fm + 'f8' \
                 }

            dtype = []
            count = 0
            element = None
            for line in header:
                if line.startswith('element'):
                    # new element definition starts
                    #  element <element-name> <number-in-file>

                    # read just completed element
                    _load_element(m, f, format, element, dtype, count)

                    # decode element header
                    props = line.split(' ')
                    element = props[1]
                    count = int(props[2])
                    dtype = []
                        
                elif count>0 and line.startswith('property'):
                    #  property <data-type> <property-name>
                    props = line.split(' ')
                    if props[1] == 'list':
                        # property list <numerical-type size.type> <numerical-type element.type> <property-name>
                        logger.error("PLY load Error: 'list' not supported.")
                        if format == 'ascii':
                            for i in range(count):
                                f.readline()
                        else:
                            return obj
                    else:
                        dtype = dtype + [ (props[-1], df[props[1]]) ]  # (name, format, shape)

            _load_element(m, f, format, element, dtype, count)
            obj._post_process_after_load()
            return obj

        else:
            logger.error("Error: incorrect file format.")
            return None


def save_scene(filename, _object):
    with open(filename, 'wb') as f:
        save_scene_stream(f, _object)


def save_scene_stream(stream, _object):
    if isinstance(_object, model.Model):
        m = _object._mesh
    elif isinstance(_object, model.Mesh):
        m = _object
    else:
        print("Unknown object type '{0}'. Unable to save".format(type(_object)))
        return

    binary = True

    if m is not None:
        frame = "ply\n"
        if binary:
            frame += "format binary_little_endian 1.0\n"
        else:
            frame += "format ascii 1.0\n"
        frame += "comment Generated by Horus / Gryphon Scan {0}\n".format(__version__)
        frame += "element vertex {0}\n".format(m.vertex_count)
        frame += "property float x\n"
        frame += "property float y\n"
        frame += "property float z\n"
        frame += "property uchar red\n"
        frame += "property uchar green\n"
        frame += "property uchar blue\n"
        frame += "property uchar scalar_Original_cloud_index\n"
        frame += "property int slice_index\n"
        frame += "property float slice_angle\n"

        if m.metadata is not None:
            metadata = pickle.dumps(m.metadata, 2)
            if binary:
                frame += "element metadata {0}\n".format(len(metadata))
                frame += "property uchar data\n"
            else:
                frame += "element metadata 1\n" # single line of data
                frame += "property uchar data\n"
        else:
            print("No metadata to save")

        frame += "element face 0\n"
        frame += "property list uchar int vertex_indices\n"

        frame += "end_header\n"

        stream.write(frame)

        if m.vertex_count > 0:
            if binary:
                for i in range(m.vertex_count):
                    stream.write(struct.pack("<fffBBBBif",
                                             m.vertexes[i, 0], m.vertexes[i, 1], m.vertexes[i, 2],
                                             m.colors[i, 0], m.colors[i, 1], m.colors[i, 2],
                                             m.vertexes_meta[i][0], m.vertexes_meta[i][1], m.vertexes_meta[i][2]))
                if m.metadata is not None:
                    stream.write(metadata)
            else:
                for i in range(m.vertex_count):
                    stream.write("{0} {1} {2} {3} {4} {5} {6} {7} {8}\n".format(
                                 m.vertexes[i, 0], m.vertexes[i, 1], m.vertexes[i, 2],
                                 m.colors[i, 0], m.colors[i, 1], m.colors[i, 2],
                                 m.vertexes_meta[i][0], m.vertexes_meta[i][1], m.vertexes_meta[i][2]))
                if m.metadata is not None:
                    stream.write("{0}\n".format(metadata))
