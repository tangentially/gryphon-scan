# This file is part of the Gryphon Scan Project
__author__ = 'Mikhail N Klimushin aka Night Gryphon <ngryph@gmail.com>'
__copyright__ = 'Copyright (C) 2019 Night Gryphon'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'


import numpy as np

from horus.util import profile
from horus.util import model
from horus.gui.engine import driver, ciclop_scan, point_cloud_roi
from horus.gui.util.custom_panels import ExpandablePanel, ComboBox, \
     CheckBox, IntTextBox, Button, FloatTextBoxArray
from horus.gui.util.gryphon_controls import DirPicker, ColorPicker


class PointCloudColor(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(
            self, parent, _("Point cloud color"), has_undo=False, has_restore=False)
        self.main = self.GetParent().GetParent().GetParent()

    def add_controls(self):
        self.add_control('texture_mode', ComboBox)
        self.add_control('point_cloud_color', ColorPicker)
        self.add_control('point_cloud_color_l', ColorPicker)
        self.add_control('point_cloud_color_r', ColorPicker)

    def update_callbacks(self):
        self.update_callback('texture_mode', lambda v: self._set_texture_mode(v) )
        self.update_callback('point_cloud_color', ciclop_scan.set_color )
        self.update_callback('point_cloud_color_l', lambda v: ciclop_scan.set_colors(0,v) )
        self.update_callback('point_cloud_color_r', lambda v: ciclop_scan.set_colors(1,v) )

    def on_selected(self):
        self.main.scene_view._view_roi = False
        self.main.scene_view.queue_refresh()
        profile.settings['current_panel_scanning'] = 'point_cloud_color'

    def _set_texture_mode(self, mode):
        ciclop_scan.set_texture_mode(mode)

        self.get_control('point_cloud_color').Hide()
        self.get_control('point_cloud_color_l').Hide()
        self.get_control('point_cloud_color_r').Hide()

        if mode == 'Flat color':
            self.get_control('point_cloud_color').Show()
        elif mode == 'Multi color':
            self.get_control('point_cloud_color_l').Show()
            self.get_control('point_cloud_color_r').Show()
        elif mode == 'Capture':
            pass
        elif mode == 'Laser BG':
            pass
        else:
            pass

        self.parent.Layout()
        self.Layout()


class Photogrammetry(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(
            self, parent, _("Photogrammetry"), has_undo=False, has_restore=False)
        self.main = self.GetParent().GetParent().GetParent()

    def add_controls(self):
        self.add_control('ph_save_enable', CheckBox)
        self.add_control('ph_save_folder', DirPicker)
        self.add_control('ph_save_divider', IntTextBox)

    def on_selected(self):
        self.main.scene_view._view_roi = False
        self.main.scene_view.queue_refresh()
        profile.settings['current_panel_scanning'] = 'photogrammetry'


class MeshCorrection(ExpandablePanel):

    def __init__(self, parent, on_selected_callback):
        ExpandablePanel.__init__(
            self, parent, _("Scan Correction"), has_undo=False, has_restore=False)
        self.main = self.GetParent().GetParent().GetParent()
        self.mesh = None
        self.offset = np.zeros((3), dtype=np.float32)

    def add_controls(self):
        self.add_control('mesh_correction_offset', FloatTextBoxArray)
        self.add_control('mesh_correction_apply', Button)
        self.add_control('mesh_correction_reset', Button)

    def update_callbacks(self):
        self.update_callback('mesh_correction_offset', lambda v: self.set_offset(v))
        self.update_callback('mesh_correction_apply', self.apply_correction)
        self.update_callback('mesh_correction_reset', self.reset_correction)

    def on_selected(self):
        profile.settings['current_panel_scanning'] = 'mesh_correction'

    def set_offset(self, v):
        #print "Set offset {0}".format(self.offset)
        self.offset = v

    def apply_correction(self):
        if self.main.scene_view._object is None or \
           not self.main.scene_view._object._is_point_cloud:
            return

        mesh = self.main.scene_view._object._mesh
        if self.mesh is None:
            self.mesh = model.Mesh().copy(mesh)

            points_l = np.array(mesh.vertexes_meta[:,1].tolist())[:,1]
            #print "points_l {0} {1}".format(points_l.shape, points_l.dtype)
            c,s = np.cos(points_l), np.sin(points_l)
            #print "c {0} {1}".format(c.shape, c.dtype)
            self.M    = np.array([ c,-s,  s,c]).T.reshape((-1,2,2))
            self.Mrev = np.array([ c, s, -s,c]).T.reshape((-1,2,2))

        #print "self.offset {0} {1}".format(self.offset.shape, self.offset.dtype)
        delta = np.full( (mesh.vertexes.shape[0],3), [ -self.offset[2], self.offset[0], -self.offset[1]])
        #print "{0} {1}".format(delta.shape, delta.dtype)
        #print "{0} {1}".format(self.Mrev[:].shape, self.Mrev[:].dtype)
        delta[:,[0,1]] =  np.einsum('ikj,ij->ik',self.Mrev, delta[:,[0,1]])
        mesh.vertexes = self.mesh.vertexes + delta.astype(np.float32)
        mesh.clear_vbo()


    def reset_correction(self):
        if self.main.scene_view._object is None or \
           not self.main.scene_view._object._is_point_cloud:
            return

        self.main.scene_view._object._mesh.clear_vbo()
        self.mesh.clear_vbo()
        self.main.scene_view._object._mesh = self.mesh
        self.mesh = None

