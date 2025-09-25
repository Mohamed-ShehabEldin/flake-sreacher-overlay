from PyQt5.QtCore import QPoint
import pyautogui
import numpy as np
#from sam_predictor import FastSAMPredictor
# from sam2_predictor import FastSAMPredictor
from PyQt5 import QtTest



class ImageFrameManager:
    def __init__(self, image_frame):
        self.image_frame = image_frame

    def get_screenshot(self):
        self.target_star_marker.move(2, 2)
        self.receiver_star_marker.move(2, 4)
        QtTest.QTest.qWait(500)
        top_left = self.image_frame.mapToGlobal(self.image_frame.pos())
        width = self.image_frame.width()
        height = self.image_frame.height()
        screenshot = pyautogui.screenshot(region=(top_left.x(), top_left.y(), width, height))
        return np.array(screenshot.convert("RGB"))