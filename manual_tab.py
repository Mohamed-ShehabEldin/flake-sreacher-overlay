from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5 import uic
import serial.tools.list_ports
from PyQt5 import QtTest

from motion_controller import MotionController, MotionWorker

CONTINUOUS_STEPS = 100


class ManualTab(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("manual_tab.ui", self)

        self.motion_controller  = None
        self.saving_folder      = ""
        self._current_worker    = None
        self._continuous_active = False
        self._continuous_fn     = None

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

        # held buttons — continuous while pressed, stop on release
        self.xpp.pressed.connect(lambda: self._start_continuous(self.motion_controller.move_x,  CONTINUOUS_STEPS, self.x_speed_bx))
        self.xmm.pressed.connect(lambda: self._start_continuous(self.motion_controller.move_x, -CONTINUOUS_STEPS, self.x_speed_bx))
        self.ypp.pressed.connect(lambda: self._start_continuous(self.motion_controller.move_y,  CONTINUOUS_STEPS, self.y_speed_bx))
        self.ymm.pressed.connect(lambda: self._start_continuous(self.motion_controller.move_y, -CONTINUOUS_STEPS, self.y_speed_bx))
        self.zpp.pressed.connect(lambda: self._start_continuous(self.motion_controller.move_z,  CONTINUOUS_STEPS, self.z_speed_bx))
        self.zmm.pressed.connect(lambda: self._start_continuous(self.motion_controller.move_z, -CONTINUOUS_STEPS, self.z_speed_bx))

        self.xpp.released.connect(self._stop_continuous)
        self.xmm.released.connect(self._stop_continuous)
        self.ypp.released.connect(self._stop_continuous)
        self.ymm.released.connect(self._stop_continuous)
        self.zpp.released.connect(self._stop_continuous)
        self.zmm.released.connect(self._stop_continuous)

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


    ######## worker helper ########

    def _run(self, fn):
        self._current_worker = MotionWorker(fn)
        self._current_worker.done.connect(self.show_coords)
        self._current_worker.start()


    ######## continuous (held) motion ########

    def _start_continuous(self, move_fn, steps, speed_bx):
        mc    = self.motion_controller
        speed = speed_bx.value()
        self._continuous_active = True
        self._continuous_fn = lambda: (mc.set_speed(speed), move_fn(steps))
        self._fire_continuous()

    def _fire_continuous(self):
        if not self._continuous_active:
            return
        self._current_worker = MotionWorker(self._continuous_fn)
        self._current_worker.done.connect(self._on_continuous_done)
        self._current_worker.start()

    def _on_continuous_done(self):
        if self._continuous_active:
            self._fire_continuous()
        else:
            self.show_coords()

    def _stop_continuous(self):
        self._continuous_active = False


    ######## single-step motion ########

    def xpf(self):
        mc = self.motion_controller
        speed, steps = self.x_speed_bx.value(), self.x_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_x(steps)))

    def xmf(self):
        mc = self.motion_controller
        speed, steps = self.x_speed_bx.value(), self.x_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_x(-steps)))

    def ypf(self):
        mc = self.motion_controller
        speed, steps = self.y_speed_bx.value(), self.y_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_y(steps)))

    def ymf(self):
        mc = self.motion_controller
        speed, steps = self.y_speed_bx.value(), self.y_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_y(-steps)))

    def zpf(self):
        mc = self.motion_controller
        speed, steps = self.z_speed_bx.value(), self.z_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_z(steps)))

    def zmf(self):
        mc = self.motion_controller
        speed, steps = self.z_speed_bx.value(), self.z_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_z(-steps)))

    def show_coords(self):
        mc = self.motion_controller
        self.coord_display.setText(f"X: {mc.get_x()}, Y: {mc.get_y()}, Z: {mc.get_z()}")


    ######## other ########

    def pick_saving_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Saving Folder")
        if folder:
            self.saving_folder = folder
            print(f"Saving folder set to: {self.saving_folder}")

    def ai_check(self):
        print("AI check — not implemented yet")
