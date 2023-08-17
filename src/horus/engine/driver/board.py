# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import time
import glob
import serial
import threading
import platform
from horus.util import profile

import logging
logger = logging.getLogger(__name__)

system = platform.system()


class WrongFirmware(Exception):

    def __init__(self):
        Exception.__init__(self, "Wrong Firmware")


class BoardNotConnected(Exception):

    def __init__(self):
        Exception.__init__(self, "Board Not Connected")


class OldFirmware(Exception):

    def __init__(self):
        Exception.__init__(self, "Old Firmware")


class Board(object):

    """Board class. For accessing to the scanner board

    Gcode commands:

        G1 Fnnn : feed rate
        G1 Xnnn : move motor
        G50     : reset origin position

        M70 Tn  : switch off laser n
        M71 Tn  : switch on laser n

        M50 Tn  : read ldr sensor

    """

    def __init__(self, parent=None, serial_name='/dev/ttyUSB0', baud_rate=115200):
        self.parent = parent
        self.serial_name = serial_name
        self.baud_rate = baud_rate
        self.unplug_callback = None

        self._serial_port = None
        self._is_connected = False
        self._laser_number = 2
        self._light = [0,0]

        self.reset_state()


    def reset_state(self):
        self._motor_enabled = False
        self._motor_position = 0
        self._motor_speed = 0
        self._motor_acceleration = 0
        self._motor_direction = 1
        self._laser_enabled = self._laser_number * [False]
        self._tries = 0  # Check if command fails

        self._light = [0,0]


    def connect(self):
        """Open serial port and perform handshake"""
        logger.info("Connecting board {0} {1}".format(self.serial_name, self.baud_rate))
        self.reset_state()
        self._is_connected = False
        try:
            self._serial_port = serial.Serial(self.serial_name, self.baud_rate, timeout=2)
            if self._serial_port.isOpen():
                self._reset()  # Force Reset and flush
                version = ""
                tic     = time.time()
                while ( ((time.time() - tic) < 1) and (profile.settings['firmware_string'] not in version) ):
                    tic     = time.time()
                    version = self._serial_port.readline()
                if profile.settings['firmware_string'] in version:
                    self.motor_speed(1)
                    self._serial_port.timeout = 0.05
                    self._is_connected = True
                    # Send init string
                    if profile.settings['init_string']:
                        self._send_command(profile.settings['init_string'].encode('ascii','ignore'))
                    # Set current position as origin
                    self.motor_speed(profile.settings['motor_speed_control'])
                    self.motor_acceleration(profile.settings['motor_acceleration_control'])
                    self.motor_invert(profile.settings['invert_motor'])
                    self.motor_reset_origin()
                    logger.info(" Done")
                else:
                    raise WrongFirmware()
            else:
                raise BoardNotConnected()
        except Exception as exception:
            logger.error("Error opening the port {0}\n".format(self.serial_name))
            self._serial_port = None
            raise exception

    def disconnect(self):
        """Close serial port"""
        if self._is_connected:
            logger.info("Disconnecting board {0}".format(self.serial_name))
            try:
                if self._serial_port is not None:
                    self.lasers_off()
                    self.motor_disable()
                    self._is_connected = False
                    self._serial_port.close()
                    del self._serial_port
            except serial.SerialException:
                logger.error("Error closing the port {0}\n".format(self.serial_name))
            logger.info(" Done")

    def set_unplug_callback(self, value):
        self.unplug_callback = value

    def motor_invert(self, value):
        if value:
            self._motor_direction = -1
        else:
            self._motor_direction = +1

    def motor_speed(self, value):
        if self._is_connected:
            if self._motor_speed != value:
                self._motor_speed = value
                self._send_command("G1F{0}".format(value))

    def motor_acceleration(self, value):
        if self._is_connected:
            if self._motor_acceleration != value:
                self._motor_acceleration = value
                self._send_command("$120={0}".format(value))

    def motor_enable(self):
        if self._is_connected:
            if not self._motor_enabled:
                self._motor_enabled = True
                # Save current speed value
                speed = self._motor_speed
                self.motor_speed(1)
                # Enable stepper motor
                self._send_command("M17")
                time.sleep(1)
                # Restore speed value
                self.motor_speed(speed)

    def motor_disable(self):
        if self._is_connected:
            if self._motor_enabled:
                self._motor_enabled = False
                self._send_command("M18")

    def motor_reset_origin(self):
        if self._is_connected:
            self._send_command("G50")
            self._motor_position = 0

    def motor_move(self, step=0, nonblocking=False, callback=None):
        if self._is_connected:
            #print("Move: "+str(step))
            self._motor_position += step * self._motor_direction
            self.send_command("G1X{0:f}".format(self._motor_position), nonblocking, callback)

    def laser_on(self, index):
        if self._is_connected:
            if not self._laser_enabled[index]:
                self._laser_enabled[index] = True
                self._send_command("M71T" + str(index + 1))

    def laser_off(self, index):
        if self._is_connected:
            if self._laser_enabled[index]:
                self._laser_enabled[index] = False
                self._send_command("M70T" + str(index + 1))

    def lasers_on(self):
        for i in range(self._laser_number):
            self.laser_on(i)

    def lasers_off(self):
        for i in range(self._laser_number):
            self.laser_off(i)

    def ldr_sensor(self, pin):
        value = self._send_command("M50T" + pin, read_lines=True).split("\n")[0]
        try:
            return int(value)
        except ValueError:
            return 0

    def send_command(self, req, nonblocking=False, callback=None, read_lines=False):
        if nonblocking:
            threading.Thread(target=self._send_command,
                             args=(req, callback, read_lines)).start()
        else:
            self._send_command(req, callback, read_lines)

    def _send_command(self, req, callback=None, read_lines=False):
        """Sends the request and returns the response"""
        ret = ''
        if self._is_connected and req != '':
            if self._serial_port is not None and self._serial_port.isOpen():
                try:
                    self._serial_port.flushInput()
                    self._serial_port.flushOutput()
                    #print("Cmd: "+req)
                    self._serial_port.write(req + "\r\n")
                    while req != '~' and req != '!' and ret == '':
                        ret = self.read(read_lines)
                        #print(ret)
                        time.sleep(0.01)
                    if ret.lower().rstrip("\r\n") != 'ok':
                        logger.warning("[ WARN board command ] '{0}' => '{1}'".format(req, ret.rstrip("\r\n")))
                    self._success()
                except:
                    if hasattr(self, '_serial_port'):
                        if callback is not None:
                            callback(ret)
                        self._fail()
        if callback is not None:
            callback(ret)
        #print("Cmd DONE: "+ret)
        return ret

    def read(self, read_lines=False):
        if read_lines:
            return ''.join(self._serial_port.readlines())
        else:
            return ''.join(self._serial_port.readline())

    def _success(self):
        self._tries = 0

    def _fail(self):
        if self._is_connected:
            logger.debug("Board fail")
            self._tries += 1
            if self._tries >= 3:
                self._tries = 0
                if self.unplug_callback is not None and \
                   self.parent is not None and \
                   not self.parent.unplugged:
                    self.parent.unplugged = True
                    self.unplug_callback()

    def _reset(self):
        self._serial_port.flushInput()
        self._serial_port.flushOutput()
        self._serial_port.write("\x18\r\n")  # Ctrl-x
        self._serial_port.readline()

    @staticmethod
    def get_serial_list():
        """Obtain list of serial devices"""
        baselist = []
        if system == 'Windows':
            import _winreg
            try:
                key = _winreg.OpenKey(
                    _winreg.HKEY_LOCAL_MACHINE, "HARDWARE\\DEVICEMAP\\SERIALCOMM")
                i = 0
                while True:
                    try:
                        values = _winreg.EnumValue(key, i)
                    except:
                        return baselist
                    if 'USBSER' in values[0] or \
                       'VCP' in values[0] or \
                       '\Device\Serial' in values[0]:
                        baselist.append(values[1])
                    i += 1
            except:
                return baselist
        else:
            for device in ['/dev/ttyACM*', '/dev/ttyUSB*', '/dev/tty.usb*', '/dev/tty.wchusb*',
                           '/dev/cu.*', '/dev/rfcomm*']:
                baselist = baselist + glob.glob(device)
        return baselist


    def set_light(self, idx, brightness):
        if not self._is_connected:
           return False

        try:
            if self._light[idx] == brightness:
                return False

            if brightness > 0:
                if brightness > 255:
                    brightness = 255
                self._light[idx] = brightness
                self.parent.board._send_command("M71T{0}F{1}".format(idx+3, brightness)) # Laser on with brightness
            else:
                self._light[idx] = 0
                self.parent.board._send_command("M70T{0}".format(idx+3)) # Laser off
            return True
        except IndexError:
            pass

        return False

    def lights_off(self):
        for i in range(len(self._light)):
            self.set_light(i,0)

