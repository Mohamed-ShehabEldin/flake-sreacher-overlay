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

class AutoScan(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("autoscan_tab.ui", self)