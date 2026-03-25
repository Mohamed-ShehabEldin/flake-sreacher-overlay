from PyQt5.QtCore import QPoint
import pyautogui
import numpy as np


class ImageFrameManager:
    def __init__(self, image_frame):
        self.image_frame = image_frame

    def get_screenshot(self):
        trim = 15
        top_left = self.image_frame.mapToGlobal(QPoint(0, 0))
        x = top_left.x() + trim
        y = top_left.y() + trim
        width  = self.image_frame.width()  - 2 * trim
        height = self.image_frame.height() - 2 * trim
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return np.array(screenshot.convert("RGB"))
