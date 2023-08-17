# -*- coding: utf-8 -*-
# This file is part of the Gryphon Scan Project

__author__ = 'Mikhail N Klimushin aka Night Gryphon <ngryph@gmail.com>'
__copyright__ = 'Copyright (C) 2019 Night Gryphon'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import types
import numpy as np
from scipy import optimize, stats, spatial

#from horus.util import model

import logging
logger = logging.getLogger(__name__)

# bin tree for atan()
class TanNode(object):
    def __init__(self, angle=0, delta=90, level = 0, bit = 0):
        self.angle = angle
        self.tan = np.tan(np.deg2rad(angle))
        self.less = None
        self.more = None
        self.bit = bit << level
        if level>0:
            self.less = TanNode(angle-delta/2, delta/2, level-1, 0)
            self.more = TanNode(angle+delta/2, delta/2, level-1, 1)

    def get(self, value):
        if self.less is not None:
            if value < self.tan:
                angle, index = self.less.get(value)
            else:
                angle, index = self.more.get(value)
            return angle, self.bit | index
        else:
            return self.angle, self.bit

class Cloud(object):
    def __init__(self, points_xyz   = np.empty((0,3), dtype=np.float32), \
                       points_l     = np.empty((0), dtype=np.float32), \
                       points_color = np.empty((0,3), dtype=np.uint8), \
                       points_rt = None, length = None):
        self.points_xyz   = np.array(points_xyz)
        if points_rt is not None:
            self.points_rt    = np.array(points_rt) # polar
        else:
            self.points_rt    = None # polar
        self.points_l     = np.array(points_l) # point scan angle
        self.points_color = np.array(points_color, dtype=np.uint8) # color
        self.M = None
        self.Mrev = None

        self.resize(length)


    def clear(self, length = 0):
        self.points_xyz = np.array([0.,0.,0.]*length)
        self.points_rt = None # polar
        self.points_l = np.array([0.]*length) # point scan angle
        self.points_color = np.array([0,0,0]*length, dtype=np.uint8) # color


    def resize(self, length):
        if length is not None:
            self.points_xyz.resize( (length,3), refcheck=False )
            self.points_l.resize( (length), refcheck=False )
            self.points_color.resize( (length,3), refcheck=False )
            if self.points_rt is not None:
                self.points_rt.resize( (length,2), refcheck=False )


    def add(self, points_xyz, points_l, points_color = np.array([0,0,0], dtype=np.uint8), points_rt = None):
        if len(points_xyz.shape) <= 1:
            # add single point
            #print "Add: {0} {1} {2} {3}".format(points_xyz, points_l, points_color, points_rt)
            if self.points_rt is not None:
                if points_rt is None:
                    points_rt = cart2pol([points_xyz[[0,1]]])
                if len(self.points_rt) > 0:
                    self.points_rt = np.append(self.points_rt, [points_rt], axis=0)
                else:
                    self.points_rt = [points_rt]

            self.points_xyz   = np.append(self.points_xyz,   [points_xyz], axis=0)
            self.points_l     = np.append(self.points_l,     [points_l], axis=0)
            self.points_color = np.append(self.points_color, [points_color], axis=0)
        else:
            print("Adding {0} points".format(len(points_l)))
            # add lists
            if self.points_rt is not None:
                if points_rt is None:
                    points_rt = cart2pol(points_xyz[:,[0,1]])
                if len(self.points_rt) > 0:
                    self.points_rt = np.append(self.points_rt, points_rt, axis=0)
                else:
                    self.points_rt = points_rt

            if len(self.points_xyz) > 0:
                self.points_xyz   = np.append(self.points_xyz,   points_xyz, axis=0)
                self.points_l     = np.append(self.points_l,     points_l, axis=0)
                self.points_color = np.append(self.points_color, points_color, axis=0)
            else:
                self.points_xyz   = points_xyz
                self.points_l     = points_l
                self.points_color = points_color

        return(len(self.points_xyz)-1)

    def get_rt(self):
        if self.points_rt is None:
            self.points_rt = cart2pol(self.points_xyz[:,[0,1]])
        return self.points_rt

    def reduce_by_count(self, cnt=10):
        self.points_xyz    = mean_n(self.points_xyz, cnt)
        if self.points_rt is not None:
            self.points_rt = mean_n(self.points_rt, cnt)
        self.points_l      = cicrcmean_n(self.points_l, cnt)
        self.points_color  = mean_n(self.points_color.astype(np.uint16), cnt).astype(np.uint8)

    def make_M(self):
        c,s = np.cos(self.points_l), np.sin(self.points_l)
        self.M = np.array([c,-s,s,c]).T.reshape((-1,2,2))
        self.Mrev = np.array([c,s,-s,c]).T.reshape((-1,2,2))


class ChunksPolar(object):
    def __init__(self, width = 2., height = 2., maxvar=4., min_amount = 3):
        self.width = np.deg2rad(width)
        self.height = height
        self.maxvar = maxvar
        self.min_amount = min_amount
        self.cloud = Cloud( points_rt=np.empty((0,2), dtype=np.float32) ) # avg points for chunks
        self.src_cloud = None
        self.chunks = {}
        self.chunks_count = 0

    def put_points(self, src_cloud):
        self.src_cloud = src_cloud
        self.chunks = {}
        self.chunks_count = 0

        print("Build polar chunks")
        points_rt = np.copy(src_cloud.get_rt())

        if points_rt is not None:
            # make chunks centers
            t = np.around(points_rt[:,1]/self.width).astype(int)
            z = np.around(src_cloud.points_xyz[:,2]/self.height).astype(int)

            # glue cylinder seam. theta is in [-PI ... +PI] range. 
            # Glue +180 points to first -180 chunk.
            mx = np.around(np.pi/np.deg2rad(self.width) ).astype(int) # border index
            idx = np.where(t>=mx)
            points_rt[idx, 1] -= 2*np.pi
            t[idx] -= 2*mx

            # group points to chunks
            print("\tGrouping points")
            for _id,(_t,_z) in enumerate(zip(t, z)):
                self.chunks.setdefault(_z,{}).setdefault(_t,[]).\
                       append(_id)

            # calculate chunks parameters
            print("\tCalculating chunks")
            delete = []
            for _z,T in self.chunks.iteritems():
                # T - current horizontal slice (Thetas list)
                s = {}
                for _t,D in T.iteritems():
                    # D - current chunk point ids
                    _var = np.var(points_rt[D], axis=0)
                    _cnt = len(D)
                    if _var[0] <= self.maxvar and \
                       _cnt >= self.min_amount:
                        avg_xyz = np.mean(src_cloud.points_xyz[D], axis=0)
                        avg_c   = np.mean(src_cloud.points_color.astype(np.uint16)[D], axis=0).astype(np.uint8)
                        avg_rt  = np.array([ np.mean(points_rt[D][:,0], axis=0), stats.circmean(points_rt[D][:,1], axis=0) ])
                        avg_l   = stats.circmean(src_cloud.points_l[D], axis=0)
                        # point_xyz, point_l, point_color, point_rt
                        _id = self.cloud.add(avg_xyz, avg_l, avg_c, points_rt = avg_rt)
                        s[_t] = [_id,D]
                        self.chunks_count += 1
                # replace layer
                if len(s)>10: # minimum amount of chunks for precise align ( minimum = 2 to solve equations )
                    self.chunks[_z] = s
                    print("\tChunk z={0} - {1}".format(_z, len(s)))
                else:
                    delete.append(_z)
            for _x in delete:
                del self.chunks[_x]
        print("[Done] build {0} chunks".format(self.chunks_count))

    def get_center_vertexes(self):
        res=np.empty( (self.chunks_count,3), dtype=np.float32)
        cnt = 0
        for _z,T in iter(self.chunks.items()): # horizontal slices
            for _t,D in T.iteritems():
                #print "{0},{1}: {2}".format(_z, _t, D[0])
                #print self.cloud.points_rt[D[0]]
                res[cnt] = [self.cloud.points_rt[D[0]][0], _t*self.width, _z*self.height]
                cnt += 1

        res[:,[0,1]] = pol2cart(res[:,[0,1]])
        return res


    def intersect(self, chunksB):
        buf=[0,0]*min(self.chunks_count, chunksB.chunks_count)
        cnt = 0
        res={}
        for _z,TA in iter(self.chunks.items()): # horizontal slices
            TB = chunksB.chunks.get(_z,None)
            if TB is None:
                continue # no matching chunk in B

            for _t,DA in TA.iteritems():
                DB = TB.get(_t,None)
                if DB is None:
                    continue

                buf[cnt] = [DA[0], DB[0]]
                cnt += 1

            if cnt>0:
                res[_z] = buf[0:cnt]
                cnt = 0
        return res

    def fit_chunks(self, chunksB):
        self.cloud.make_M()
        chunksB.cloud.make_M()

        res=[]
        delta = np.array([0.,0.])
        for _z,TA in iter(self.chunks.items()): # horizontal slices
            TB = chunksB.chunks.get(_z,None)
            if TB is None:
                continue # no matching Z layer in B

            idxA = []
            for _t,D in TA.iteritems():
                idxA.append(D[0])

            idxB = []
            for _t,D in TB.iteritems():
                idxB.append(D[0])

            print("Layer {0}: {1} vs {2} points".format(_z, len(idxA), len(idxB)))
            delta = fit_clouds( self.cloud.points_xyz[idxA][:,[0,1]], self.cloud.Mrev[idxA],
                                chunksB.cloud.points_xyz[idxB][:,[0,1]], chunksB.cloud.Mrev[idxB], delta )
            res += [delta+[_z*self.height]]
            print(">>>>>>>>> {0} <<<<<<<<<<<".format(delta + [_z * self.height]))

        return np.array(res)


class ChunksCubic(object):
    def __init__(self, width = 2., height = 2., min_amount = 3):
        self.width = width
        self.height = height
        self.min_amount = min_amount
        self.cloud = Cloud() # avg points for chunks
        self.src_cloud = None
        self.chunks = {}
        self.chunks_count = 0

    def put_points(self, src_cloud):
        self.src_cloud = src_cloud
        self.chunks = {}
        self.chunks_count = 0

        print("Build cubic chunks")
        # make chunks centers
        print(src_cloud.points_xyz.shape)
        xyz = np.around(src_cloud.points_xyz/np.array([self.width, self.width, self.height])).astype(int)
        print(xyz.shape)

        # group points to chunks
        print("\tGrouping points")
        for _id,(_x,_y,_z) in enumerate(xyz):
            self.chunks.setdefault(_z,{}).setdefault(_y,{}).setdefault(_x,[]).\
                   append(_id)
        
        # calculate chunks parameters
        print("\tCalculating chunks")
        delete = []
        for _z,Y in self.chunks.iteritems():
            # T - current horizontal slice (Thetas list)
            yy = {}
            for _y,X in Y.iteritems():
                xx = {}
                for _x,D in X.iteritems():
                    # D - current chunk point ids
                    if len(D) >= self.min_amount:
                        #print "1 -> {0}".format(len(D))
                        avg_xyz =        np.mean(src_cloud.points_xyz[D], axis=0)
                        avg_l   = stats.circmean(src_cloud.points_l[D], axis=0)
                        avg_c   =        np.mean(src_cloud.points_color.astype(np.uint32)[D], axis=0).astype(np.uint8)
                        #print "2"
                        # point_xyz, point_l, point_color, point_rt
                        _id = self.cloud.add(avg_xyz, avg_l, avg_c)
                        xx[_x] = [_id,D]
                        self.chunks_count += 1
                # replace layer
                if len(xx)>0:
                  yy[_y] = xx
                #print "\t\tChunk y={0} -> {1}".format(_y, len(xx))
            # replace layer
            if len(yy)>0:
                self.chunks[_z] = yy
                print("\tChunk z={0} -> {1}".format(_z, len(yy)))
            else:
                delete.append(_z)
        for _z in delete:
            del self.chunks[_z]
        print("[Done] build {0} chunks".format(self.chunks_count))

    def get_center_vertexes(self):
        res=np.empty( (self.chunks_count,3), dtype=np.float32)
        cnt = 0
        for _z,Y in iter(self.chunks.items()): # horizontal slices
            for _y,X in iter(Y.items()):
                for _x,D in iter(X.items()):
                    res[cnt] = [_x*self.width, _y*self.width, _z*self.height]
                    cnt += 1

        return res


    def intersect(self, chunksB):
        print("Intersect chunks {0} vs {1}".format(self.chunks_count, chunksB.chunks_count))
        buf=[0,0] * min(self.chunks_count, chunksB.chunks_count)
        cnt = 0
        res = {}
        for _z,YA in iter(self.chunks.items()): # horizontal slices
            YB = chunksB.chunks.get(_z,None)
            if YB is None:
                continue # no matching chunk in B

            for _y,XA in YA.iteritems():
                XB = YB.get(_y,None)
                if XB is None:
                    continue

                for _x,DA in XA.iteritems():
                    DB = XB.get(_x,None)
                    if DB is None:
                        continue

                    buf[cnt] = [DA[0], DB[0]]
                    cnt += 1
            if cnt>0:
                print("\tChunk z={0} -> {1}".format(_z, cnt))
                res[_z] = buf[0:cnt]
                cnt = 0
        return res


class MeshTools(object):
    def __init__(self, mesh = None):
        self.mesh = mesh

    def get_laser_clouds(self):
        print(spatial.KDTree)
        print("Splitting mesh by laser id")
        res = {}
        for p in zip(self.mesh.vertexes, self.mesh.colors, self.mesh.vertexes_meta)[0:self.mesh.vertex_count]:
            c = res.setdefault(p[2][0], Cloud())
            c.add(p[0], p[2][1][1], p[1])

        return res


    def get_laser_clouds2(self):
        # with preallocate array
        print("Splitting mesh by laser id")
        idx = {}
        res = {}
        for p in zip(self.mesh.vertexes, self.mesh.colors, self.mesh.vertexes_meta)[0:self.mesh.vertex_count]:
            c = res.setdefault(p[2][0], Cloud(length = self.mesh.vertex_count))
            i = idx.setdefault(p[2][0], 0)
            c.points_xyz[i] = p[0]
            c.points_l[i] = p[2][1][1]
            c.points_color[i] = p[1]
            idx[p[2][0]] += 1

        for i,p in iter(idx.items()):
            res[i].resize(p)
        return res


    def have_slices(self):
        if len(self.mesh.vertexes)<=0:
            return None
        return self.mesh.vertexes_meta[0][1] is not None


    def reconstruct_slices(self, step = None):
        print("Reconstruct slices")
        # step - scanning step in radians
        #step = np.deg2rad(0.9)
        first_laser = np.min(self.mesh.vertexes_meta[:,0])
        cur_slice = 0
        prev_laser = first_laser
        prev_z = 65535
        for m,v in zip(self.mesh.vertexes_meta,self.mesh.vertexes):
            if m[0] != prev_laser and m[0] == first_laser:
                cur_slice += 1
            elif m[0] == prev_laser and v[2]-5 > prev_z:
                cur_slice += 1

            if step is None:
                m[1] = (cur_slice, 0)
            else:
                m[1] = (cur_slice, cur_slice*step)
            prev_laser = m[0]
            prev_z = v[2]

        if step is None and cur_slice > 0:
            step = 2*np.pi/(cur_slice)
            #for m,c in zip(self.vertexes_meta, self.colors):
            for m in self.mesh.vertexes_meta:
                m[1] = (m[1][0], m[1][0]*step)
                #c[0] = int(m[1][0]*255/801)
                #c[1] = c[0]
                #c[2] = c[0]

        logger.info("{0} Slices reconstructed. Angle: {1} deg".format(cur_slice, np.rad2deg(step)))
            


class CloudTools(object):
    def __init__(self, mesh = None):
        self.set_mesh(mesh)

    def set_mesh(self, mesh):
        self.mesh = mesh
        # clone source mesh
        self.from_mesh()

        self.radial = None
        self.chunks = None

    def from_mesh(self):
        if self.mesh is not None:
            self.vertexes     = self.mesh.vertexes
            self.colors       = self.mesh.colors
            self.normal       = self.mesh.normal
            self.vertex_count = self.mesh.vertex_count
            self.vertexes_meta   = self.mesh.vertexes_meta
        else:
            self.vertexes     = None
            self.colors       = None
            self.normal       = None
            self.vertex_count = 0
            self.vertexes_meta   = None

    def to_mesh(self):
        if self.mesh is None:
            return

        self.mesh.vertexes = self.vertexes
        self.mesh.vertex_count = len(self.vertexes)
        self.mesh.colors = self.colors
        self.mesh.normal = self.normal
        self.mesh.vertexes_meta = self.vertexes_meta

    # Unwrap point cloud to cylindrical coords
    def make_radial(self):
        if self.vertex_count > 0:
            print("Make polar coords cache")
            #[x,y,z] = m.vertexes.T
            r = np.linalg.norm(self.vertexes[:,0:2], axis=1)
            t = np.arctan2(self.vertexes[:,1],self.vertexes[:,0])
            #z = self.vertexes[:,2]
            self.radial = np.array(zip(r,t))
    
    def unwrap_mesh(self, width = 360., scale_z=1.):
        if self.mesh is None:
            return

        if self.radial is None:
            self.make_radial()

        if self.radial is not None:
            vertexes = np.array(zip(self.radial[:,0], (self.radial[:,1])*width/2/np.pi, self.vertexes[:,2]*scale_z))
            self.mesh.vertexes = vertexes
            self.mesh.vertex_count = len(vertexes)
            self.mesh.colors       = self.colors
            self.mesh.normal       = self.normal
            self.mesh.vertexes_meta   = self.vertexes_meta


    def flatten_mesh(self, width = 360., scale_z=1.):
        if self.mesh is None:
            return

        if self.radial is None:
            self.make_radial()

        if not self.have_slices():
            self.reconstruct_slices()

        #ll = self.vertexes_meta[:,0] # laser
        #col = np.array( [ ll*255, ll*255, ll*255 ], dtype=np.uint8).T

        l = np.array(self.vertexes_meta[:,1].tolist(),dtype=np.float32)[:,1] # angle
        #l = l*0xFF/2/np.pi 
        #col = np.array( [ l, l, l ], dtype=np.uint8).T

        if self.radial is not None:
            x = self.radial[:,0]*np.cos(self.radial[:,1]+l)
            y = self.radial[:,0]*np.sin(self.radial[:,1]+l)
            vertexes = np.array(zip(x,y,self.vertexes[:,2]))
            self.mesh.vertexes = vertexes
            self.mesh.vertex_count = len(vertexes)
            self.mesh.colors       = self.colors
            self.mesh.normal       = self.normal
            self.mesh.vertexes_meta   = self.vertexes_meta


    def have_slices(self):
        if len(self.vertexes)<=0:
            return None
        return self.vertexes_meta[0][1] is not None

    def reconstruct_slices(self, step = None):
        print("Reconstruct slices")
        # step - scanning step in radians
        #step = np.deg2rad(0.9)
        first_laser = np.min(self.vertexes_meta[:,0])
        cur_slice = 0
        prev_laser = first_laser
        prev_z = 65535
        for m,v in zip(self.vertexes_meta,self.vertexes):
            if m[0] != prev_laser and m[0] == first_laser:
                cur_slice += 1
            elif m[0] == prev_laser and v[2]-5 > prev_z:
                cur_slice += 1

            if step is None:
                m[1] = (cur_slice, 0)
            else:
                m[1] = (cur_slice, cur_slice*step)
            prev_laser = m[0]
            prev_z = v[2]

        if step is None and cur_slice > 0:
            step = 2*np.pi/(cur_slice)
            #for m,c in zip(self.vertexes_meta, self.colors):
            for m in self.vertexes_meta:
                m[1] = (m[1][0], m[1][0]*step)
                #c[0] = int(m[1][0]*255/801)
                #c[1] = c[0]
                #c[2] = c[0]

        logger.info("{0} Slices reconstructed. Angle: {1} deg".format(cur_slice, np.rad2deg(step)))
            
    def get_corrected_vertices(self, delta = [0,0]):
        if self.radial is None:
            self.make_radial()

        assert self.radial is not None, "No input vertices (self.radial == None)"
        print("Get corrected {0}, {1} points".format(delta, len(self.vertexes)))

        l = np.array(self.vertexes_meta[:,1].tolist(),dtype=np.float32)[:,1] # angle
        res = np.copy(self.vertexes)  # keep original data intact
        res[:,1] += l
        res = pol2cart(res)
        res[:] += np.array(delta)
        res = cart2pol(res)
        res[:,1] -= l
        res = pol2cart(res)
        res = np.insert(res, 2, self.vertexes[:,2], axis=1)

        return np.array(res, dtype=np.float32)

    def build_chunks(self, width = 2., height = 2., maxvar=4., min_amount = 3):
        print("Build chunks")
        if self.radial is None:
            self.make_radial()

        if not self.have_slices():
            self.reconstruct_slices()

        if self.radial is not None:
            t = np.around(self.radial[:,1]/np.deg2rad(width)).astype(int)
            z = np.around(self.vertexes[:,2]/height).astype(int)
            
            # glue cylinder seam. theta is in [-PI ... +PI] range. 
            # Glue +180 points to first -180 chunk.
            rad = np.copy(self.radial) # keep original data intact
            mx = np.around(np.pi/np.deg2rad(width) ).astype(int) # border index
            idx = np.where(t>=mx)
            rad[idx, 1] -= 2*np.pi
            t[idx] = -mx
            #t[ t >= mx ] = -mx

            # group points to chunks
            self.chunks = {}
            print("Grouping points")
            #for _id,(_r,_t,_z,_c,_m) in enumerate(zip(self.radial, t, z, \
            for _id,(_r,_t,_z,_c,_m) in enumerate(zip(rad, t, z, \
                 self.colors, self.vertexes_meta)): # radial, chunk_theta, chunk_z, color, laser num
                self.chunks.setdefault(_m[0],{'width': width, 'height': height}).\
                              setdefault(_z,{}).\
                                setdefault(_t,[]).\
                       append([ _id,_r,np.array(_c, np.uint16),_m[1][1] ]) # id, [radial], [color], slice_l
        
            # calculate chunks parameters
            print("Calculating chunks")
            for _l,C in self.chunks.iteritems():
                # C - chunk for current laser cloud
                delete = []
                for _z,T in C.iteritems():
                    # T - current horizontal slice (Thetas list)
                    #print "T: {0}:{1}".format(_z,T)
                    if not isinstance(_z, (int, long)):
                        continue # metadata

                    s = {}
                    for _t,D in T.iteritems():
                        # D - current chunk Data
                        # id, [radial], [color], slice_l
                        D = np.array(D)
                        _var = np.var(D[:,1], axis=0)
                        _cnt = len(D)
                        if _var[0] <= maxvar and \
                           _cnt >= min_amount:
                            if _var[1]>np.deg2rad(10):
                                print("Out points {0}".format(np.rad2deg(np.array(D[:, 1].tolist())[:, 1])))
                            avg_r = np.mean(D[:,1], axis=0)
                            avg_c = np.mean(D[:,2], axis=0).astype(np.uint8)
                            #avg_l = np.mean(D[:,3], axis=0)
                            avg_l = stats.circmean(np.array(D[:,3], dtype=np.float32), axis=0)
                            _id = D[:,0]
                            # [radial], [color], slice_l, variance, count, [orig index]
                            s[_t] = np.array([avg_r, avg_c, avg_l, _var, _cnt, _id])
                            #s[_t] = np.array([avg_r, avg_c, D[0,3], _var, _cnt, _id])
                            #s[_t] = np.array([D[0,1], avg_c, D[0,3], _var, _cnt, _id])
                    # replace layer
                    if len(s)>10: # minimum amount of chunks for precise align ( minimum = 2 to solve equations )
                        C[_z] = s
                        print("Chunk z={0} - {1}".format(_z, len(s)))
                    else:
                        delete.append(_z)
                for _x in delete:
                    del C[_x]
            print("Done build chunks")

    @staticmethod
    def get_chunk_vertexes(chunk, delta=[0, 0]):
        if chunk is None:
            return np.array([]),np.array([])

        print("Retreive chunk vertices")
        width = np.deg2rad(chunk['width'])
        height = chunk['height']
        vertexes = []
        colors = []
        for _z,T in chunk.iteritems():
            # T - current horizontal slice
            if isinstance(_z, (int, long)):
                #print "Slice {0}: {1}".format(_z,T)
                for _t,D in T.iteritems():
                    # current chunk
                    # [radial], [color], slice_l, variance, count, [orig index]
                    #print "D: {0}".format(D)
                    #vertexes.append( [D[0][0]*np.cos(D[0][1]), D[0][0]*np.sin(D[0][1]), _z*height] ) # points
                    #vertexes.append( [D[0][0]*np.cos(_t*width), D[0][0]*np.sin(_t*width), _z*height] ) # chunk center
                    #vertexes.append( [D[0][0]*np.cos(D[0][1]+D[2]), D[0][0]*np.sin(D[0][1]+D[2]), _z*height] ) # point reverted
                    #vertexes.append( [D[0][0]*np.cos(_t*width+D[2]), D[0][0]*np.sin(_t*width+D[2]), _z*height] ) # chunk center reverted

                    x = D[0][0]*np.cos(D[0][1]+D[2])+delta[0]
                    y = D[0][0]*np.sin(D[0][1]+D[2])+delta[1]
                    X = x * np.cos(-D[2]) - y * np.sin(-D[2])
                    Y = y * np.cos(-D[2]) + x * np.sin(-D[2])
                    vertexes.append( [X, Y, _z*height] ) # points

                    colors.append( D[1] )
                    #if np.rad2deg(_t*width) < 0:
                    #    vertexes.append( [D[0][0]*np.cos(D[0][1]+D[2]), D[0][0]*np.sin(D[0][1]+D[2]), _z*height] )
                    #    colors.append( D[1] )

        return np.array(vertexes, dtype=np.float32), np.array(colors, dtype=np.uint8)


    def adjust_chunks(self, chunkA, chunkB):
        print("Adjust chunks")

        assert chunkA['width'] == chunkB['width'] and \
            chunkA['height'] == chunkB['height'], \
            "Compared chunks must have same chunk size"

        width  = chunkA['width']
        height = chunkA['height']
        res = [] #{'width': width, 'height': height}
                    
        # prepare point indexes
        ls = np.array(self.vertexes_meta[:,1].tolist())[:,1] # turntable L
        delta = [0] # [0,0]
        for _z,TA in chunkA.iteritems(): # horizontal slices
            if not isinstance(_z, (int, long)):
                continue # skip metadata

            TB = chunkB.get(_z,None)
            if TB is None:
                continue # no matching chunk in B

            # TA,TB - valid horizontal slices to compare
            pA = []
            pB = []
            lAB = []
            Cn = []
            for _t,DA in TA.iteritems():
                # current chunk
                DB = TB.get(_t,None)
                if DB is None:
                    continue # no matching chunk in B

                # D: [radial], [color], slice_l, variance, count, [orig index]
                pA.append([ DA[0][0], DA[0][1]+DA[2] ])
                pB.append([ DB[0][0], DB[0][1]+DB[2] ])
                lAB.append(np.abs(DA[2]-DB[2]))

                # scaling correction 
                # assume points are at their correct angular places: A to B angle corresponds actual angle
                Ax = DA[0][0] * np.cos(DA[0][1]+DA[2])
                Ay = DA[0][0] * np.sin(DA[0][1]+DA[2])

                Bx = DB[0][0] * np.cos(DB[0][1]+DB[2])
                By = DB[0][0] * np.sin(DB[0][1]+DB[2])

                l = 2*np.tan(np.abs(DA[2]-DB[2])/2)

                Px = (Ax+Bx)/2
                Py = (Ay+By)/2

                Vx = (By-Ay)/l
                Vy = (Ax-Bx)/l

                C = np.array([[Px+Vx,Py+Vy],[Px-Vx,Py-Vy]], dtype=np.float32)

                L = np.linalg.norm(C, axis=1)
                if (L[0]<L[1]):
                    Cn.append(C[0])
                    #print "{0}: {1}".format(_z, C[0])
                else:
                    Cn.append(C[1])
                    #print "{0}: {1}".format(_z, C[1])
            if len(pA) > 10: # amount of corresponding chunks
                # minimum 2 required to solve. but for better precision use 10+. Also look at "build_chunks"
                C = np.mean(Cn, axis=0)
                #print "C: {0} => {1}".format(Cn, C)
                pA = pol2cart(pA)+C
                pB = pol2cart(pB)+C

                #v = np.append(pA,pB,axis=0)
                #v = np.array(pA,dtype=np.float32)
                #v = np.insert(v, 2, _z*height, axis=1)
                #res += v.tolist()

                #delta = fit_correction(pA, pB, [0,0])
                delta = fit_correction(pA, pB, lAB, delta)
                #res[_z] = [delta[0],delta[1], _z]
                res.append([C[0]+delta[0], C[1], _z*height] )
            #res.append(np.append(np.mean(Cn, axis=0), [_z*height]) )
        return np.array(res, dtype=np.float32)


    '''
    def unwrap_cloud_image(_object, width = 360, height = 1024, scale_z = 1):
        image = None
        t, r, z, color, cloud_index = unwrap_cloud(_object)
        if t is not None:
            t = np.around((t+np.pi)*(width/2/np.pi)).astype(int)
            z = np.around(z*scale_z).astype(int)
            ind = np.argwhere(z>0 and z<height)
    
            image = np.empty((width,height), dtype=list)
            map(lambda x: map(lambda y: list(), x), z[:])
             d.setdefault(6,[]).append(5)
            #np.put(image, (t[ind], z[ind]), r[ind])
            image[ t[ind], z[ind] ] = zip(r[ind], colors[ind], vertexes_meta[ind])
    
            # m.colors[i, 0], m.colors[i, 1], m.colors[i, 2], m.vertexes_meta[i]))
        return image
    '''


# =======================
def cart2pol(points):
    # [ x, y ]
    points = np.array(points)
    r = np.linalg.norm(points, axis=1)
    t = np.arctan2(points[:,1], points[:,0])
    return np.array(zip(r, t), dtype=np.float32)

def pol2cart(radial):
    # [ radius, theta ]
    radial = np.array(radial)
    x = radial[:,0] * np.cos(radial[:,1])
    y = radial[:,0] * np.sin(radial[:,1])
    return np.array(zip(x, y), dtype=np.float32)

# ----------------------------------
def mean_n(arr, cnt):
    return np.mean( arr.reshape((-1,cnt)+arr.shape[1:]), axis=1)

def cicrcmean_n(arr, cnt):
    return stats.circmean( arr.reshape((-1,cnt)+arr.shape[1:]), axis=1)

# ----------------- R Mat -------------
def apply_mat_arr(mat, arr):
    # mat - array  of matrices N x [2x2]
    # arr - array of vectors N x [2]
    #print "Apply matrix M: {0} to V: {1}".format(mat.shape, arr.shape)
    s = arr.shape[1]
    assert s == 2 or s == 3, "N x 2D or N x 3D vectors required"
    assert mat.shape[1:] == (s,s), "N x [{0}x{0}] matrices required".format(s)
    assert mat.shape[0] == arr.shape[0], "Number of matrices should match number of vectors"
    return np.einsum('ikj,ij->ik',mat,arr)


# https://math.stackexchange.com/questions/1365622/adding-two-polar-vectors

# ======================================================    
# calculate table center correction for set of points
# assume Z change is negligible

# https://scipy.github.io/old-wiki/pages/Cookbook/Least_Squares_Circle.html
# https://stackoverflow.com/questions/6949370/scipy-leastsq-dfun-usage
# https://algowiki-project.org/ru/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%9D%D1%8C%D1%8E%D1%82%D0%BE%D0%BD%D0%B0_%D0%B4%D0%BB%D1%8F_%D1%81%D0%B8%D1%81%D1%82%D0%B5%D0%BC_%D0%BD%D0%B5%D0%BB%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D1%8B%D1%85_%D1%83%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B9

def risiduals_fit_correction(parameters,pA,pB,lAB):
    # TODO add correction vector rotation
    '''
    rM = (np.linalg.norm( pA[:], axis=1 ) + np.linalg.norm( pB[:], axis=1 ) )/2
    A = pA[:] + parameters
    B = pB[:] + parameters
    rA = np.linalg.norm( A, axis=1 )
    rB = np.linalg.norm( B, axis=1 )
    #l = pA[:,0]**2 + pA[:,1]**2 + pB[:,0]**2 + pB[:,1]**2 - 2*np.dot(pA,pB.T)
    #l2 = A[:,0]**2 +  A[:,1]**2 +  B[:,0]**2 +  B[:,1]**2 - 2*np.dot(pA,pB.T)
    l1 = np.dot(A,B.T)
    l2 = rA*rB*np.cos(lAB)
    #return np.append(np.abs(rA-rM), np.abs(rB-rM))
    return np.append(rA-rB, l1-l2)
    '''
    A = pA[:] + [parameters[0],0]
    B = pB[:] + [parameters[0],0]
    #rA = np.linalg.norm( A, axis=1 )
    #rB = np.linalg.norm( B, axis=1 )
    #print rA-rB
    d = np.linalg.norm( A-B, axis=1 )
    #print(d)
    return d

def fit_correction(pA, pB, lAB, prev = [0]): #[0,0]):
    # pA,pB - cartensian coords of points reverted to capture position
    '''
    V = prev
    print "Fit A {0}, B {1}, lAB {2}".format(len(pA),len(pB),len(lAB))
    offset, ier = optimize.leastsq(risiduals_fit_correction, V, args=( (pA, pB, lAB) ))
    print "Fit result: {0}  ier={1}  delta_r={2}".format(offset, ier, np.mean(risiduals_fit_correction(offset,pA,pB,lAB)) )
    #print np.round( np.linalg.norm( pA[:], axis=1 ), 3)
    #print np.round( np.linalg.norm( pA[:] + offset, axis=1 ), 3)
    #print np.round( np.linalg.norm( pB[:] + offset, axis=1 ), 3)
    #print np.round( risiduals_fit_correction(offset,pA,pB), 3)
    #print np.round( cart2pol(pA)[:,0], 3)
    #print np.round( cart2pol(pA[:] + offset)[:,0], 3)

    return offset    
    '''
    # pA,pB - cartensian coords of points reverted to capture position
    V = prev
    #print "Fit A {0}, B {1}, lAB {2}".format(len(pA),len(pB),len(lAB))
    offset, _, _, _, ier = optimize.leastsq(risiduals_fit_correction, V, args=( (pA, pB, lAB) ))
    print("Fit result: {0}  ier={1}  delta_r={2}".format(offset, ier,
                                                         np.mean(risiduals_fit_correction(offset, pA, pB, lAB))))
    #print np.round( np.linalg.norm( pA[:], axis=1 ), 3)
    #print np.round( np.linalg.norm( pA[:] + offset, axis=1 ), 3)
    #print np.round( np.linalg.norm( pB[:] + offset, axis=1 ), 3)
    #print np.round( risiduals_fit_correction(offset,pA,pB), 3)
    #print np.round( cart2pol(pA)[:,0], 3)
    #print np.round( cart2pol(pA[:] + offset)[:,0], 3)

    return offset    









'''
def calc_R(parameters, points):
    # points = (r, theta - l_slice)
    #    r, theta - point coords in model space; l_slice - turntable coords angle offset (capture angle)
    # Parameters: 
    #    d,gamma - platform center offset
    d,gamma = parameters 

    #return [sqrt(r**2 + d**2 - 2*r*d*np.cos(theta - l_slice - gamma)) for r,theta,l_slice in points] 
    return np.sqrt(points[0]**2 + d**2 - 2*points[0]*d*np.cos(points[1] - gamma)) 

def risiduals_fit_correction(parameters, points):
    d,gamma = parameters 

    Ri = calc_R(parameters, points)
    return Ri - Ri.mean()

#def d_fit_correction(parameters, points):
#    return [ -2, 2*d-2*r*np.cos(theta - l_slice - gamma), 
    
def fit_correction(data):
    # data: r, theta - l_slice

    print "\nFit data: {0}".format(data)
    print np.rad2deg(data[0][1]-data[1][1])
    # R,d,gamma
    V = 0, 0
    offset, ier = optimize.leastsq(risiduals_fit_correction, V, args=(data))
    print "Result: {0}  ier={1}".format(offset, ier)

    return offset
'''
def risiduals_fit_clouds(V, PA, MAneg, PB, MBneg):
    A = PA+ apply_mat_arr(MAneg, np.full((PA.shape[0],2), V))
    B = PB+ apply_mat_arr(MBneg, np.full((PB.shape[0],2), V))
    #print "A: {0}\nB:{1} {2}".format(A.shape, B.shape, B[20]-PB[20])
    print("\n\tA: {0}\tB:{1}".format(A[10] - PA[10], B[10] - PB[10]))
    tA = spatial.KDTree(A)
    tB = spatial.KDTree(B)
    #print len(tA.query_pairs(10))
    diff = tA.sparse_distance_matrix(tB, 99999999)
    res = diff.mean() # TODO mean of minimums
    print("\t{0} -> {1}".format(V, res))
    #return [res,res]
    return diff.sum(axis=0)


def fit_clouds(PA, MAneg, PB, MBneg, prev = np.array([0.,0.])):
    V = prev
    offset, _, _, _, ier = optimize.leastsq(risiduals_fit_clouds, V, args=( (PA, MAneg, PB, MBneg) ))
    print("Fit result: {0}  ier={1}".format(offset, ier))
    #res = optimize.least_squares(risiduals_fit_clouds, V, args=( (PA, MAneg, PB, MBneg) ), bounds = [(-15,-15),(15,15)] )
    #offset = res.x
    #print "-----------------------------\nFit result: {0}\n===============================\n".format(res)

    return offset


