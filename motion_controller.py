from PyQt5.QtCore import QPoint
import pyautogui
import numpy as np
from PyQt5 import QtTest
import serial



class MotionController:
    def __init__(self, com_port):
        self.com_port = com_port
        self.motion_controller = None  # Will be initialized upon connection
        self.absolute_x = 0
        self.absolute_y = 0
        self.absolute_z = 0
        self.ser = None
        self.connect_device()

    def connect_device(self):
        # connect to the arduino device
        try:
            self.ser = serial.Serial(self.com_port, baudrate=2000000, timeout=1)
            QtTest.QTest.qWait(2000)
            return True
        except Exception as e:
            print(e)
            return False
        
    def move_x(self, step=1, speed=100):
        '''Move the X axis by a specified number of steps at a given speed.
        step: Number of steps to move (positive for forward, negative for backward).'''
        if self.ser:
            command = f'to be written\n'.encode()
            self.ser.write(command)
            absolute_x += step

    def move_y(self, step=1, speed=100):
        '''Move the Y axis by a specified number of steps at a given speed.
        step: Number of steps to move (positive for forward, negative for backward).'''

        if self.ser:
            command = f'to be written\n'.encode()
            self.ser.write(command)
            absolute_y += step

    def move_z(self, step=1, speed=100):
        '''Move the Z axis by a specified number of steps at a given speed.
        step: Number of steps to move (positive for forward, negative for backward).'''

        if self.ser:
            command = f'to be written\n'.encode()
            self.ser.write(command)
            absolute_z += step

    def get_x(self):
        return self.absolute_x
    def get_y(self):
        return self.absolute_y
    def get_z(self):
        return self.absolute_z

    def disconnect(self):
        if self.ser:
            self.ser.close()
            self.ser = None