from PyQt5.QtWidgets import QWidget
from PyQt5 import uic
#this file should make the elements in the "settings_tabWidget" functional
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QSlider, QLineEdit, QHBoxLayout, QPushButton, QRadioButton, QMessageBox, QFileDialog, QTabWidget, QLabel, QComboBox, QCheckBox,QSizePolicy
from PyQt5 import uic, QtWidgets
import serial.tools.list_ports
import time
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPainter, QBrush, QPen
from PyQt5 import QtTest

import argparse
import json
import os

import cv2
import numpy as np
# start by importing the necessary packages
import matplotlib.pyplot as plt
import json
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage
import pyautogui
import sys
from PyQt5.QtWidgets import QDialog, QMainWindow, QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider,QLineEdit, QHBoxLayout, QPushButton, QRadioButton, QMessageBox, QFileDialog, QTabWidget, QLabel, QComboBox,QCheckBox
from PyQt5 import uic
import serial.tools.list_ports
import time
#from moving_fun import fun_testEachAxisMoveLoop
from PyQt5 import QtTest

import sys
import time


from window_interaction_handler import WindowInteractionHandler
from motion_controller import MotionController

class ManualTab(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("manual_tab.ui", self)

        #T for temprature, M for motion

        ports = [port.device for port in serial.tools.list_ports.comports()]

        ####### Motion cpntroller init #######
        self.combo_connect_M.addItems(ports)
        self.push_connect_M.clicked.connect(self.connect_M_device)
        self.motion_controller = None  # won't be None once connected
        self.xp.pressed.connect(self.xpf)
        self.xm.pressed.connect(self.xmf)
        self.yp.pressed.connect(self.ypf)
        self.ym.pressed.connect(self.ymf)
        self.zp.pressed.connect(self.zpf)
        self.zm.pressed.connect(self.zmf)


        #self.push_show_coords.pressed.connect(self.show_coords)

        QtTest.QTest.qWait(200)
        
    def show_coords(self):
        if self.motion_controller:
            x = self.motion_controller.get_x()
            y = self.motion_controller.get_y()
            z = self.motion_controller.get_z()
            self.coord_display.setText(f"X: {x}, Y: {y}, Z: {z}")
        else:
            self.coord_display.setText("Not connected")

    ######### Motion Controller ########
    def connect_M_device(self):
        comPort=self.combo_connect_M.currentText()
        self.MotionController=MotionController(comPort)
        self.status=self.MotionController.connect_device()
        if not self.status:
            print(f"Failed to initialize motion control card on {comPort}!")
            self.MController_status.setText("Error!")
            sys.exit(1)
        print(f"Successfully connected to motion control card on {comPort}")
        self.MController_status.setText("Connected")


    def xpf(self):
        speed=self.x_speed_bx.value()
        step=self.x_step_bx.value()
        self.motion_controller.move_x(step,speed)
        self.show_coords()

    def xmf(self):
        speed=self.x_speed_bx.value()
        step=self.x_step_bx.value()
        self.motion_controller.move_x(-step,speed)
        self.show_coords()
    def ypf(self):
        speed=self.y_speed_bx.value()
        step=self.y_step_bx.value()
        self.motion_controller.move_y(step,speed)
        self.show_coords()
    def ymf(self):
        speed=self.y_speed_bx.value()
        step=self.y_step_bx.value()
        self.motion_controller.move_y(-step,speed)
        self.show_coords()
    def zpf(self):
        speed=self.z_speed_bx.value()
        step=self.z_step_bx.value()
        self.motion_controller.move_y(step,speed)
        self.show_coords()

    def zmf(self):
        speed=self.z_speed_bx.value()
        step=self.z_step_bx.value()
        self.motion_controller.move_y(-step,speed)
        self.show_coords()