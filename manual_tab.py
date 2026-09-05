from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5 import uic
import serial.tools.list_ports
from collections import deque

from motion_controller import MotionController, MotionWorker

CONTINUOUS_STEPS = 100


class ManualTab(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("manual_tab.ui", self)

        self.motion_controller  = None
        self._current_worker    = None
        self._workers           = []   # strong refs — prevent GC while thread runs
        self._move_queue        = deque()
        self._worker_busy       = False
        self._continuous_active = False
        self._continuous_fn     = None
        self._held_keys         = set()
        self._arrows_running    = False
        self._scan_active       = False

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
        self.xpp.pressed.connect(lambda: self._start_continuous('x',  CONTINUOUS_STEPS, self.x_speed_bx))
        self.xmm.pressed.connect(lambda: self._start_continuous('x', -CONTINUOUS_STEPS, self.x_speed_bx))
        self.ypp.pressed.connect(lambda: self._start_continuous('y',  CONTINUOUS_STEPS, self.y_speed_bx))
        self.ymm.pressed.connect(lambda: self._start_continuous('y', -CONTINUOUS_STEPS, self.y_speed_bx))

        self.xpp.released.connect(self._stop_continuous)
        self.xmm.released.connect(self._stop_continuous)
        self.ypp.released.connect(self._stop_continuous)
        self.ymm.released.connect(self._stop_continuous)
        self.zpp.released.connect(self._stop_continuous)
        self.zmm.released.connect(self._stop_continuous)

        # move-to absolute position
        self.move_to_x_btn.clicked.connect(self.move_to_x)
        self.move_to_y_btn.clicked.connect(self.move_to_y)
        self.move_to_z_btn.clicked.connect(self.move_to_z)

        self.setFocusPolicy(Qt.StrongFocus)
        self._disable_z_controls()
        self.show_coords()

    def _disable_z_controls(self):
        message = "Z axis unavailable: current firmware supports X and Y only."
        for widget in (
            self.zp, self.zm, self.zpp, self.zmm,
            self.z_speed_bx, self.z_angle_bx,
            self.move_to_z_btn, self.move_to_z_spinbx,
        ):
            widget.setEnabled(False)
            widget.setToolTip(message)

    def _xy_motion_widgets(self):
        return (
            self.xp, self.xm, self.yp, self.ym,
            self.xpp, self.xmm, self.ypp, self.ymm,
            self.move_to_x_btn, self.move_to_y_btn,
            self.move_to_x_spinbx, self.move_to_y_spinbx,
            self.x_speed_bx, self.y_speed_bx,
            self.x_angle_bx, self.y_angle_bx,
            self.arrows_ctrl_chkBx,
        )

    def set_scan_active(self, active):
        self._scan_active = active
        for widget in self._xy_motion_widgets():
            widget.setEnabled(not active)
        self.combo_connect_M.setEnabled(not active)
        if active:
            self.push_connect_M.setText("Disconnect")
        else:
            self.push_connect_M.setText("Reconnect" if self._controller_ready() else "Connect")

    def has_active_motion(self):
        return bool(
            self._workers or self._move_queue or self._worker_busy
            or self._continuous_active or self._arrows_running
        )

    def _controller_ready(self):
        mc = self.motion_controller
        return mc is not None and mc.is_connected() and mc.is_position_valid()

    def _require_controller(self):
        if self._scan_active:
            self.MController_status.setText("Auto scan owns stage")
            return None
        if not self._controller_ready():
            self.MController_status.setText("Connect stage first — position unknown")
            self.show_coords()
            return None
        return self.motion_controller


    ######## arrow key control ########

    def keyPressEvent(self, event):
        if not self.arrows_ctrl_chkBx.isChecked() or event.isAutoRepeat():
            return super().keyPressEvent(event)
        if not self._controller_ready() or self._scan_active:
            return super().keyPressEvent(event)
        key = event.key()
        if key in (Qt.Key_Right, Qt.Key_Left, Qt.Key_Up, Qt.Key_Down):
            self._held_keys.add(key)
            if not self._arrows_running:
                self._arrows_running = True
                self._fire_arrows()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if not self.arrows_ctrl_chkBx.isChecked() or event.isAutoRepeat():
            return super().keyReleaseEvent(event)
        key = event.key()
        if key in (Qt.Key_Right, Qt.Key_Left, Qt.Key_Up, Qt.Key_Down):
            self._held_keys.discard(key)
            if not self._held_keys:
                self._arrows_running = False
        else:
            super().keyReleaseEvent(event)

    def _arrow_fn(self):
        mc = self.motion_controller
        if mc is None or not mc.is_connected() or not mc.is_position_valid():
            return
        x_steps = 0
        y_steps = 0
        if Qt.Key_Right in self._held_keys: x_steps += CONTINUOUS_STEPS
        if Qt.Key_Left  in self._held_keys: x_steps -= CONTINUOUS_STEPS
        if Qt.Key_Up    in self._held_keys: y_steps += CONTINUOUS_STEPS
        if Qt.Key_Down  in self._held_keys: y_steps -= CONTINUOUS_STEPS
        if x_steps != 0:
            mc.move_x(x_steps, speed=self.x_speed_bx.value(), owner=self)
        if y_steps != 0:
            mc.move_y(y_steps, speed=self.y_speed_bx.value(), owner=self)

    def _fire_arrows(self):
        if not self._arrows_running or not self._held_keys:
            self._arrows_running = False
            return
        mc = self.motion_controller
        if mc is None or not mc.is_connected() or not mc.is_position_valid():
            self._arrows_running = False
            return
        worker = MotionWorker(self._arrow_fn)
        self._workers.append(worker)
        worker.done.connect(self._on_arrows_done)
        worker.failed.connect(self._on_motion_failed)
        worker.finished.connect(lambda w=worker: self._workers.remove(w) if w in self._workers else None)
        worker.start()

    def _on_arrows_done(self):
        if self._arrows_running and self._held_keys:
            self._fire_arrows()
        else:
            self._arrows_running = False
            self.show_coords()

    ######## connection ########

    def connect_M_device(self):
        if self._scan_active:
            if self.motion_controller:
                self.motion_controller.disconnect()
            self.MController_status.setText("Disconnected — scan stopping; position unknown")
            self.show_coords()
            return

        comPort = self.combo_connect_M.currentText()
        if not comPort:
            self.MController_status.setText("No serial port selected")
            return
        if self.has_active_motion():
            self.MController_status.setText("Wait for current motion before reconnecting")
            return
        if self.motion_controller:
            self.motion_controller.disconnect()
        self.motion_controller = MotionController(comPort)
        if not self.motion_controller.is_connected():
            print(f"Failed to connect on {comPort}!")
            self.MController_status.setText("Connection failed — position unknown")
            self.show_coords()
            return
        print(f"Connected on {comPort}")
        self.MController_status.setText("Connected")
        self.push_connect_M.setText("Reconnect")
        self.show_coords()


    ######## worker helper ########

    def _run(self, fn):
        self._move_queue.append(fn)
        self._dispatch()

    def _dispatch(self):
        if self._worker_busy or not self._move_queue:
            return
        fn = self._move_queue.popleft()
        self._worker_busy = True
        worker = MotionWorker(fn)
        self._workers.append(worker)
        worker.done.connect(self._on_move_done)
        worker.failed.connect(self._on_motion_failed)
        worker.finished.connect(lambda w=worker: self._workers.remove(w) if w in self._workers else None)
        self._current_worker = worker
        worker.start()

    def _on_move_done(self):
        self._worker_busy = False
        self.show_coords()
        self._dispatch()


    ######## continuous (held) motion ########

    def _start_continuous(self, axis, steps, speed_bx):
        mc = self._require_controller()
        if mc is None:
            return
        speed = speed_bx.value()
        self._continuous_active = True
        move_fn = mc.move_x if axis == 'x' else mc.move_y
        self._continuous_fn = lambda: move_fn(steps, speed=speed, owner=self)
        self._fire_continuous()

    def _fire_continuous(self):
        if not self._continuous_active:
            return
        if not self._controller_ready() or self._scan_active:
            self._continuous_active = False
            return
        worker = MotionWorker(self._continuous_fn)
        self._workers.append(worker)
        worker.done.connect(self._on_continuous_done)
        worker.failed.connect(self._on_motion_failed)
        worker.finished.connect(lambda w=worker: self._workers.remove(w) if w in self._workers else None)
        self._current_worker = worker
        worker.start()

    def _on_continuous_done(self):
        if self._continuous_active:
            self._fire_continuous()
        else:
            self.show_coords()

    def _stop_continuous(self):
        self._continuous_active = False

    def _on_motion_failed(self, message):
        self._continuous_active = False
        self._arrows_running = False
        self._held_keys.clear()
        self._move_queue.clear()
        self.MController_status.setText(f"Motion error — {message}")
        self.show_coords()


    ######## single-step motion ########

    def xpf(self):
        mc = self._require_controller()
        if mc is None:
            return
        speed, steps = self.x_speed_bx.value(), self.x_angle_bx.value()
        self._run(lambda: mc.move_x(steps, speed=speed, owner=self))

    def xmf(self):
        mc = self._require_controller()
        if mc is None:
            return
        speed, steps = self.x_speed_bx.value(), self.x_angle_bx.value()
        self._run(lambda: mc.move_x(-steps, speed=speed, owner=self))

    def ypf(self):
        mc = self._require_controller()
        if mc is None:
            return
        speed, steps = self.y_speed_bx.value(), self.y_angle_bx.value()
        self._run(lambda: mc.move_y(steps, speed=speed, owner=self))

    def ymf(self):
        mc = self._require_controller()
        if mc is None:
            return
        speed, steps = self.y_speed_bx.value(), self.y_angle_bx.value()
        self._run(lambda: mc.move_y(-steps, speed=speed, owner=self))

    def zpf(self):
        self.MController_status.setText("Z axis unavailable in current firmware")

    def zmf(self):
        self.MController_status.setText("Z axis unavailable in current firmware")

    def move_to_x(self):
        mc = self._require_controller()
        if mc is None:
            return
        steps = self.move_to_x_spinbx.value() - mc.get_x()
        if steps != 0:
            speed = self.x_speed_bx.value()
            self._run(lambda: mc.move_x(steps, speed=speed, owner=self))

    def move_to_y(self):
        mc = self._require_controller()
        if mc is None:
            return
        steps = self.move_to_y_spinbx.value() - mc.get_y()
        if steps != 0:
            speed = self.y_speed_bx.value()
            self._run(lambda: mc.move_y(steps, speed=speed, owner=self))

    def move_to_z(self):
        self.MController_status.setText("Z axis unavailable in current firmware")

    def show_coords(self):
        mc = self.motion_controller
        if mc is None:
            self.coord_display.setText("X: ?, Y: ?, Z: unavailable — not connected")
        elif not mc.is_position_valid():
            self.coord_display.setText("X: ?, Y: ?, Z: unavailable — POSITION UNKNOWN")
        else:
            self.coord_display.setText(f"X: {mc.get_x()}, Y: {mc.get_y()}, Z: unavailable")
