# This file is part of the Gryphon Scan Project
__author__ = 'Mikhail N Klimushin aka Night Gryphon <ngryph@gmail.com>'
__copyright__ = 'Copyright (C) 2019 Night Gryphon'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core
import types
import struct

from horus.util import profile
from horus.gui.util.custom_panels import ControlPanel


class Header(ControlPanel):

    def __init__(self, parent, name, tooltip=None):
        #ControlPanel.__init__(self, parent, name, engine_callback)
        wx.Panel.__init__(self, parent)
        self.name = name
        if tooltip:
            self.SetToolTip(wx.ToolTip(tooltip))

        self.control = None
        self.undo_values = []
        self.engine_callback = None
        self.append_undo_callback = None
        self.release_undo_callback = None

        # Elements
        self.label = wx.StaticText(self, label=name)
        font = self.label.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        #font = wx.Font(18, wx.DECORATIVE, wx.NORMAL, wx.BOLD)
        self.label.SetFont(font)
        self.line = wx.StaticLine(self)

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.label, 0, wx.ALL ^ wx.TOP ^ wx.BOTTOM | wx.EXPAND, 10)
        vbox.Add(self.line, 1, wx.ALL | wx.EXPAND, 8)
        self.SetSizer(vbox)
        self.Layout()

    def append_undo(self):
        pass

    def release_undo(self):
        pass

    def release_restore(self):
        pass

    def undo(self):
        pass

    def reset_profile(self):
        pass

    def update_from_profile(self):
        pass

    def update_to_profile(self, value):
        pass

    def set_engine(self, value):
        pass


class DirPicker(ControlPanel):

    def __init__(self, parent, name, engine_callback=None):
        def _bound_SetValue(self, value):
            self.SetPath(value)
        
        def _bound_GetValue(self, value):
            return self.GetPath()

        ControlPanel.__init__(self, parent, name, engine_callback)

        # Elements
        label = wx.StaticText(self, size=(140, -1), label=_(self.setting._label))
        self.control = wx.DirPickerCtrl(self, size=(120, -1), style= wx.DIRP_USE_TEXTCTRL ) #| wx.DIRP_SMALL ) # | wx.DIRP_DIR_MUST_EXIST)
        self.control.SetPath(profile.settings[self.name])
        self.control.SetValue = types.MethodType(_bound_SetValue, self.control)
        self.control.GetValue = types.MethodType(_bound_GetValue, self.control)

        # Layout
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(label, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.hbox.AddStretchSpacer()
        self.hbox.Add(self.control, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        self.SetSizer(self.hbox)
        self.Layout()

        # Events
        self.control.Bind(wx.EVT_DIRPICKER_CHANGED, self._on_dir_changed)

    def _on_dir_changed(self, event):
        value = self.control.GetPath()
        self.update_to_profile(value)
        self.set_engine(value)
        self.release_restore()


class ColorPicker(ControlPanel):

    def __init__(self, parent, name, engine_callback=None):
        ControlPanel.__init__(self, parent, name, engine_callback)
        # Elements
        label = wx.StaticText(self, size=(130, -1), label=_(self.setting._label))
        self.control = wx.Button(self, label="", size=(150, -1), style=wx.BORDER_NONE)
        self.update_from_profile()

        # Layout
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(label, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        hbox.AddStretchSpacer()
        hbox.Add(self.control, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        self.SetSizer(hbox)
        self.Layout()

        # Events
        self.control.Bind(wx.EVT_BUTTON, self._on_btn_click)


    def update_from_profile(self):
        value = self.decode_color(self.setting.value)
        if self.control is not None:
            self.set_control_value(value)
            self.set_engine(value)

    @staticmethod
    def decode_color(value):
        if isinstance(value, str):
            ret = struct.unpack('BBB', bytes.fromhex(value))
        elif isinstance(value, (tuple,list)) and \
             len(value) == 3 and \
             all(isinstance(x, int) for x in value):
           ret = value
        else:
           ret = (0,0,0)
        return ret


    def update_to_profile(self, value):
        if issubclass(self.setting._type, str):
            profile.settings[self.name] = str("".join(map(chr, value)).encode('hex'))
        elif issubclass(self.setting._type, list):
            profile.settings[self.name] = value
        elif issubclass(self.setting._type, tuple):
            profile.settings[self.name] = tuple(value)


    def set_control_value(self, value):
        self.control.SetBackgroundColour(wx.Colour(value[0] & 0xFF, value[1] & 0xFF, value[2] & 0xFF))
        self.control.SetLabel( "#{0}\n{1} {2} {3}".format("".join(map(chr, value)).encode('hex'), \
                  value[0] & 0xFF, value[1] & 0xFF, value[2] & 0xFF ) )

    def _on_btn_click(self, event):
        v = self.pick_color()
        if v is not None:
            self.set_control_value(v)
            self.update_to_profile(v)
            self.set_engine(v)

    def pick_color(self):
        ret = None
        data = wx.ColourData()
        data.SetColour(self.setting.value)
        dialog = wx.ColourDialog(self, data)
        dialog.GetColourData().SetChooseFull(True)
        if dialog.ShowModal() == wx.ID_OK:
            data = dialog.GetColourData()
            ret = list(data.GetColour().Get())
        dialog.Destroy()
        return ret

