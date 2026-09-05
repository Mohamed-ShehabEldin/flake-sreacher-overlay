from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5 import uic
import os
import cv2
import time
from collections import deque

from motion_controller import MotionError, MotionWorker

CONTINUOUS_STEPS = 100
CONTINUOUS_MS    = 50
SETTLE_S         = 0.5   # wait after each motor move before screenshot


def _unique_path(folder, stem):
    """Return folder/stem.png, incrementing stem_N if the file exists."""
    path = os.path.join(folder, f"{stem}.png")
    n = 1
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{n}.png")
        n += 1
    return path


class ScanWorker(QThread):
    step_done  = pyqtSignal(dict)   # info dict for UI update
    stage_failed = pyqtSignal(str)

    def __init__(self, mc, image_frame_manager, pipeline,
                 fast_axis, fast_n, fast_angle, fast_speed,
                 slow_n, slow_angle, slow_speed,
                 save_all, save_folder,
                 ratio, batch_size, radius, zigzag, owner):
        super().__init__()
        self.mc                  = mc
        self.image_frame_manager = image_frame_manager
        self.pipeline            = pipeline
        self.fast_axis           = fast_axis
        self.fast_n              = fast_n
        self.fast_angle          = fast_angle
        self.fast_speed          = fast_speed
        self.slow_n              = slow_n
        self.slow_angle          = slow_angle
        self.slow_speed          = slow_speed
        self.save_all            = save_all
        self.save_folder         = save_folder
        self.ratio               = ratio
        self.batch_size          = batch_size
        self.radius              = radius
        self.zigzag              = zigzag
        self.owner               = owner
        self._stop               = False

    def stop(self):
        self._stop = True

    def _move_fast(self, mc, steps):
        if self.fast_axis == 'x':
            mc.move_x(steps, speed=self.fast_speed, owner=self.owner)
        else:
            mc.move_y(steps, speed=self.fast_speed, owner=self.owner)

    def _move_slow(self, mc):
        if self.fast_axis == 'x':
            mc.move_y(self.slow_angle, speed=self.slow_speed, owner=self.owner)
        else:
            mc.move_x(self.slow_angle, speed=self.slow_speed, owner=self.owner)

    def run(self):
        mc    = self.mc
        total = self.slow_n * self.fast_n
        done  = 0

        try:
            for slow_i in range(self.slow_n):
                if self._stop:
                    break

                direction = 1 if (not self.zigzag or slow_i % 2 == 0) else -1

                for fast_j in range(self.fast_n):
                    if self._stop:
                        break

                    # screenshot at current position first (no pre-move)
                    time.sleep(SETTLE_S)
                    if not mc.is_connected() or not mc.is_position_valid():
                        raise MotionError("Stage disconnected or its position became unknown.")
                    x, y, z = mc.get_x(), mc.get_y(), mc.get_z()

                    try:
                        rgb = self.image_frame_manager.get_screenshot()
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        self.step_done.emit({'error': str(e), 'done': done, 'total': total,
                                             'slow_i': slow_i, 'fast_j': fast_j,
                                             'x': x, 'y': y, 'z': z})
                        done += 1
                        if fast_j < self.fast_n - 1:
                            self._move_fast(mc, direction * self.fast_angle)
                        continue

                    # AI inference — always run if model is loaded (to get flake_size)
                    flake_found = False
                    flake_size  = 0
                    if self.pipeline._model is not None:
                        try:
                            _, _, stats = self.pipeline.test(
                                bgr,
                                ratio=self.ratio,
                                batch_size=self.batch_size,
                                radius=self.radius,
                            )
                            flake_size  = stats['filtered']
                            flake_found = flake_size > 0
                        except Exception as e:
                            self.step_done.emit({'error': f"Inference: {e}", 'done': done,
                                                  'total': total, 'slow_i': slow_i,
                                                  'fast_j': fast_j, 'x': x, 'y': y, 'z': z})

                    if self.save_all or flake_found:
                        stem = f"s{flake_size:03d}_x{x}_y{y}_z{z}"
                        path = _unique_path(self.save_folder, stem)
                        cv2.imwrite(path, bgr)

                    done += 1
                    self.step_done.emit({
                        'done': done, 'total': total,
                        'slow_i': slow_i, 'fast_j': fast_j,
                        'x': x, 'y': y, 'z': z,
                        'flake_found': flake_found, 'flake_size': flake_size,
                    })

                    # move to next position (not after the last frame in the row)
                    if fast_j < self.fast_n - 1:
                        self._move_fast(mc, direction * self.fast_angle)

                if self._stop:
                    break

                if slow_i < self.slow_n - 1:
                    self._move_slow(mc)
                    if not self.zigzag:
                        # return fast axis to row start
                        self._move_fast(mc, -(self.fast_n - 1) * self.fast_angle)
        except MotionError as e:
            self.stage_failed.emit(str(e))


class AutoScan(QWidget):
    def __init__(self, manual_tab, image_frame_manager, a_eye_tab):
        super().__init__()
        uic.loadUi("autoscan_tab.ui", self)

        self.manual_tab          = manual_tab
        self.image_frame_manager = image_frame_manager
        self.a_eye_tab           = a_eye_tab
        self.worker              = None
        self._workers            = []   # strong refs — prevent GC while thread runs
        self._move_queue         = deque()
        self._worker_busy        = False
        self._scan_owner         = None
        self._scan_error         = None

        # defaults
        self.save_relevant_rad.setChecked(True)
        self.fast_x_rad.setChecked(True)

        # timer for live coord display
        self.coord_timer = QTimer()
        self.coord_timer.timeout.connect(self._update_coords)
        self.coord_timer.start(500)
        self._update_coords()

        # worker-based continuous motion state
        self._current_worker    = None
        self._continuous_active = False
        self._continuous_fn     = None

        # saving folder
        self.saving_folder_btn.clicked.connect(self.pick_save_folder)

        # single-step
        self.xp.pressed.connect(self.xpf)
        self.xm.pressed.connect(self.xmf)
        self.yp.pressed.connect(self.ypf)
        self.ym.pressed.connect(self.ymf)

        # held (continuous)
        self.xpp.pressed.connect(lambda: self._start_continuous('x',  CONTINUOUS_STEPS, self.x_speed_bx))
        self.xmm.pressed.connect(lambda: self._start_continuous('x', -CONTINUOUS_STEPS, self.x_speed_bx))
        self.ypp.pressed.connect(lambda: self._start_continuous('y',  CONTINUOUS_STEPS, self.y_speed_bx))
        self.ymm.pressed.connect(lambda: self._start_continuous('y', -CONTINUOUS_STEPS, self.y_speed_bx))
        self.xpp.released.connect(self._stop_continuous)
        self.xmm.released.connect(self._stop_continuous)
        self.ypp.released.connect(self._stop_continuous)
        self.ymm.released.connect(self._stop_continuous)

        # multi-step (xppp/xmmm = X multiple, yppp/ymmm = Y multiple)
        self.xppp.clicked.connect(self.xpppf)
        self.xmmm.clicked.connect(self.xmmmf)
        self.yppp.clicked.connect(self.ypppf)
        self.ymmm.clicked.connect(self.ymmmf)

        # scan
        self.start_btn.clicked.connect(self.start_scan)
        self.stop_btn.clicked.connect(self.stop_scan)

    def mc(self):
        return self.manual_tab.motion_controller

    def _controller_ready(self):
        mc = self.mc()
        return mc is not None and mc.is_connected() and mc.is_position_valid()

    def _require_controller(self):
        if not self._controller_ready():
            self.scan_info.setText("Connect stage first — position unknown.")
            self._update_coords()
            return None
        return self.mc()

    def has_active_jogging(self):
        return bool(
            self._workers or self._move_queue or self._worker_busy
            or self._continuous_active
        )

    def _set_jogging_enabled(self, enabled):
        for widget in (
            self.xp, self.xm, self.yp, self.ym,
            self.xpp, self.xmm, self.ypp, self.ymm,
            self.xppp, self.xmmm, self.yppp, self.ymmm,
            self.x_speed_bx, self.y_speed_bx,
            self.x_angle_bx, self.y_angle_bx,
            self.x_multible, self.y_multible,
        ):
            widget.setEnabled(enabled)

    # ── folder ─────────────────────────────────────────────────────────────

    def pick_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select save folder")
        if folder:
            self.saving_folder_lineEdit.setText(folder)

    # ── worker helper ───────────────────────────────────────────────────────

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

    # ── continuous ──────────────────────────────────────────────────────────

    def _start_continuous(self, axis, steps, speed_bx):
        mc = self._require_controller()
        if mc is None or self.worker is not None:
            return
        speed = speed_bx.value()
        self._continuous_active = True
        move_fn = mc.move_x if axis == 'x' else mc.move_y
        self._continuous_fn = lambda: move_fn(steps, speed=speed, owner=self)
        self._fire_continuous()

    def _fire_continuous(self):
        if not self._continuous_active:
            return
        mc = self.mc()
        if mc is None or not mc.is_connected() or not mc.is_position_valid() or self.worker is not None:
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
        self._move_queue.clear()
        self.scan_info.setText(f"Motion error — {message}")
        self._update_coords()

    # ── single-step ─────────────────────────────────────────────────────────

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

    # ── multi-step ──────────────────────────────────────────────────────────

    def xpppf(self):
        mc = self._require_controller()
        if mc is None:
            return
        speed, steps = self.x_speed_bx.value(), self.x_multible.value() * self.x_angle_bx.value()
        self._run(lambda: mc.move_x(steps, speed=speed, owner=self))

    def xmmmf(self):
        mc = self._require_controller()
        if mc is None:
            return
        speed, steps = self.x_speed_bx.value(), self.x_multible.value() * self.x_angle_bx.value()
        self._run(lambda: mc.move_x(-steps, speed=speed, owner=self))

    def ypppf(self):
        mc = self._require_controller()
        if mc is None:
            return
        speed, steps = self.y_speed_bx.value(), self.y_multible.value() * self.y_angle_bx.value()
        self._run(lambda: mc.move_y(steps, speed=speed, owner=self))

    def ymmmf(self):
        mc = self._require_controller()
        if mc is None:
            return
        speed, steps = self.y_speed_bx.value(), self.y_multible.value() * self.y_angle_bx.value()
        self._run(lambda: mc.move_y(-steps, speed=speed, owner=self))

    def _update_coords(self):
        mc = self.mc()
        if mc is None:
            self.coord_display.setText("X: ?, Y: ?, Z: unavailable — not connected")
        elif not mc.is_position_valid():
            self.coord_display.setText("X: ?, Y: ?, Z: unavailable — POSITION UNKNOWN")
        else:
            self.coord_display.setText(f"X: {mc.get_x()}, Y: {mc.get_y()}, Z: unavailable")

    def show_coords(self):
        self._update_coords()

    # ── scan ────────────────────────────────────────────────────────────────

    def start_scan(self):
        mc = self._require_controller()
        if mc is None:
            print("[AutoScan] No live motion controller — connect from the Manual tab first.")
            return
        if self.worker is not None:
            self.scan_info.setText("A scan is already running.")
            return
        if self.manual_tab.has_active_motion() or self.has_active_jogging():
            self.scan_info.setText("Wait for current manual motion to finish before scanning.")
            return
        save_folder = self.saving_folder_lineEdit.text().strip()
        if not save_folder:
            print("[AutoScan] Please select a save folder first.")
            return

        fast_axis = 'x' if self.fast_x_rad.isChecked() else 'y'
        save_all  = self.save_all_rad.isChecked()

        if not save_all and self.a_eye_tab.pipeline._model is None:
            print("[AutoScan] No model loaded in A-Eye tab — switching to save all frames.")
            save_all = True

        if fast_axis == 'x':
            fast_n, fast_angle, fast_speed = self.x_multible.value(), self.x_angle_bx.value(), self.x_speed_bx.value()
            slow_n, slow_angle, slow_speed = self.y_multible.value(), self.y_angle_bx.value(), self.y_speed_bx.value()
        else:
            fast_n, fast_angle, fast_speed = self.y_multible.value(), self.y_angle_bx.value(), self.y_speed_bx.value()
            slow_n, slow_angle, slow_speed = self.x_multible.value(), self.x_angle_bx.value(), self.x_speed_bx.value()

        # read inference params from A-Eye tab
        ratio      = self.a_eye_tab.ratio_spin.value()
        batch_size = self.a_eye_tab.pred_batch_size_spin.value()
        radius     = self.a_eye_tab.radius_spin.value()

        zigzag = self.zigizag_chkbx.isChecked()

        owner = object()
        if not mc.acquire_exclusive(owner):
            self.scan_info.setText("Stage is busy or its position is unknown.")
            return

        self._scan_owner = owner
        self._scan_error = None
        self.worker = ScanWorker(
            mc, self.image_frame_manager, self.a_eye_tab.pipeline,
            fast_axis, fast_n, fast_angle, fast_speed,
            slow_n, slow_angle, slow_speed,
            save_all, save_folder,
            ratio, batch_size, radius, zigzag, owner,
        )
        self.worker.step_done.connect(self.on_step_done)
        self.worker.stage_failed.connect(self.on_stage_failed)
        self.worker.finished.connect(self.on_finished)

        self.manual_tab.set_scan_active(True)
        self._set_jogging_enabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()
        print(f"[AutoScan] Started — {slow_n}×{fast_n} grid, fast={fast_axis}, save={'all' if save_all else 'detected'}")

    def stop_scan(self):
        if self.worker:
            self.worker.stop()
            self.scan_info.setText("Stopping after the current operation...")
            print("[AutoScan] Stop requested.")

    def on_stage_failed(self, message):
        self._scan_error = message
        self.scan_info.setText(f"Scan aborted — stage error: {message}")
        self.manual_tab.MController_status.setText(f"Stage error — {message}")
        self.manual_tab.show_coords()
        print(f"[AutoScan] Scan aborted — stage error: {message}")

    def on_step_done(self, info):
        if 'error' in info:
            self.scan_info.setText(f"Error: {info['error']}")
            return
        x, y, z = info['x'], info['y'], info['z']
        self.coord_display.setText(f"X: {x}, Y: {y}, Z: unavailable")
        flake_txt = f"flake=YES ({info['flake_size']}pts)" if info['flake_found'] else "flake=no"
        self.scan_info.setText(
            f"Step {info['done']}/{info['total']}  s={info['slow_i']} f={info['fast_j']}  {flake_txt}"
        )

    def on_finished(self):
        mc = self.mc()
        if mc is not None and self._scan_owner is not None:
            mc.release_exclusive(self._scan_owner)
        self._scan_owner = None
        self.worker = None
        self.manual_tab.set_scan_active(False)
        self._set_jogging_enabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self._scan_error is None:
            self.scan_info.setText("Scan finished.")
            print("[AutoScan] Scan finished.")
