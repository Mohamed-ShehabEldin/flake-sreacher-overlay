from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5 import uic
import serial.tools.list_ports
from PyQt5 import QtTest

from motion_controller import MotionController

CONTINUOUS_STEPS = 100   # steps sent per timer tick for held buttons
CONTINUOUS_MS    = 50    # timer interval in ms


class ManualTab(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("manual_tab.ui", self)

        self.motion_controller = None
        self.saving_folder = ""

        # timer for continuous (held) buttons
        self.continuous_timer = QTimer()
        self.continuous_timer.timeout.connect(self.continuous_move)
        self.continuous_fn = None   # set when a held button is pressed

        # populate COM ports and connect the connect button
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.combo_connect_M.addItems(ports)
        self.push_connect_M.clicked.connect(self.connect_M_device)

        # single-step buttons
        self.xp.pressed.connect(self.xpf)
        self.xm.pressed.connect(self.xmf)
        self.yp.pressed.connect(self.ypf)
        self.ym.pressed.connect(self.ymf)
        self.zp.pressed.connect(self.zpf)
        self.zm.pressed.connect(self.zmf)

        # held buttons — start moving on press, stop on release
        self.xpp.pressed.connect(lambda: self.start_continuous(self.motion_controller.move_x,  CONTINUOUS_STEPS, self.x_speed_bx))
        self.xmm.pressed.connect(lambda: self.start_continuous(self.motion_controller.move_x, -CONTINUOUS_STEPS, self.x_speed_bx))
        self.ypp.pressed.connect(lambda: self.start_continuous(self.motion_controller.move_y,  CONTINUOUS_STEPS, self.y_speed_bx))
        self.ymm.pressed.connect(lambda: self.start_continuous(self.motion_controller.move_y, -CONTINUOUS_STEPS, self.y_speed_bx))
        self.zpp.pressed.connect(lambda: self.start_continuous(self.motion_controller.move_z,  CONTINUOUS_STEPS, self.z_speed_bx))
        self.zmm.pressed.connect(lambda: self.start_continuous(self.motion_controller.move_z, -CONTINUOUS_STEPS, self.z_speed_bx))

        self.xpp.released.connect(self.stop_continuous)
        self.xmm.released.connect(self.stop_continuous)
        self.ypp.released.connect(self.stop_continuous)
        self.ymm.released.connect(self.stop_continuous)
        self.zpp.released.connect(self.stop_continuous)
        self.zmm.released.connect(self.stop_continuous)

        # other buttons
        self.saving_folder_3.clicked.connect(self.pick_saving_folder)
        self.saving_folder_4.clicked.connect(self.ai_check)

        QtTest.QTest.qWait(200)


    ######## connection ########

    def connect_M_device(self):
        comPort = self.combo_connect_M.currentText()
        self.motion_controller = MotionController(comPort)
        if not self.motion_controller.ser:
            print(f"Failed to connect on {comPort}!")
            self.MController_status.setText("Error!")
            return
        print(f"Connected on {comPort}")
        self.MController_status.setText("Connected")


    ######## continuous (held) motion ########

    def start_continuous(self, move_fn, steps, speed_bx):
        self.motion_controller.set_speed(speed_bx.value())
        self.continuous_fn = lambda: move_fn(steps)
        self.continuous_fn()                        # move immediately on first press
        self.continuous_timer.start(CONTINUOUS_MS)

    def stop_continuous(self):
        self.continuous_timer.stop()
        self.continuous_fn = None
        self.show_coords()

    def continuous_move(self):
        if self.continuous_fn:
            self.continuous_fn()


    ######## motion buttons ########

    def xpf(self):
        speed = self.x_speed_bx.value()
        steps = self.x_angle_bx.value()
        self.motion_controller.set_speed(speed)
        self.motion_controller.move_x(steps)
        self.show_coords()

    def xmf(self):
        speed = self.x_speed_bx.value()
        steps = self.x_angle_bx.value()
        self.motion_controller.set_speed(speed)
        self.motion_controller.move_x(-steps)
        self.show_coords()

    def ypf(self):
        speed = self.y_speed_bx.value()
        steps = self.y_angle_bx.value()
        self.motion_controller.set_speed(speed)
        self.motion_controller.move_y(steps)
        self.show_coords()

    def ymf(self):
        speed = self.y_speed_bx.value()
        steps = self.y_angle_bx.value()
        self.motion_controller.set_speed(speed)
        self.motion_controller.move_y(-steps)
        self.show_coords()

    def zpf(self):
        speed = self.z_speed_bx.value()
        steps = self.z_angle_bx.value()
        self.motion_controller.set_speed(speed)
        self.motion_controller.move_z(steps)
        self.show_coords()

    def zmf(self):
        speed = self.z_speed_bx.value()
        steps = self.z_angle_bx.value()
        self.motion_controller.set_speed(speed)
        self.motion_controller.move_z(-steps)
        self.show_coords()

    def show_coords(self):
        x = self.motion_controller.get_x()
        y = self.motion_controller.get_y()
        z = self.motion_controller.get_z()
        self.coord_display.setText(f"X: {x}, Y: {y}, Z: {z}")


    ######## other buttons ########

    def pick_saving_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Saving Folder")
        if folder:
            self.saving_folder = folder
            print(f"Saving folder set to: {self.saving_folder}")

    def ai_check(self):
        print("AI check — not implemented yet")
