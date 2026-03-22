from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5 import uic
import os
import cv2
import time
from collections import deque

from motion_controller import MotionWorker

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
    finished   = pyqtSignal()

    def __init__(self, mc, image_frame_manager, pipeline,
                 fast_axis, fast_n, fast_angle, fast_speed,
                 slow_n, slow_angle, slow_speed,
                 save_all, save_folder,
                 ratio, batch_size, radius):
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
        self._stop               = False

    def stop(self):
        self._stop = True

    def run(self):
        mc    = self.mc
        total = self.slow_n * self.fast_n
        done  = 0

        for slow_i in range(self.slow_n):
            if self._stop:
                break

            direction = 1 if slow_i % 2 == 0 else -1   # serpentine

            for fast_j in range(self.fast_n):
                if self._stop:
                    break

                # move fast axis
                if self.fast_axis == 'x':
                    mc.set_speed(self.fast_speed)
                    mc.move_x(direction * self.fast_angle)
                else:
                    mc.set_speed(self.fast_speed)
                    mc.move_y(direction * self.fast_angle)

                time.sleep(SETTLE_S)

                x, y, z = mc.get_x(), mc.get_y(), mc.get_z()

                # screenshot
                try:
                    rgb = self.image_frame_manager.get_screenshot()
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    self.step_done.emit({'error': str(e), 'done': done, 'total': total,
                                         'slow_i': slow_i, 'fast_j': fast_j,
                                         'x': x, 'y': y, 'z': z})
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

                # save
                # print("save all status: ", self.save_all, ", flake found status: ", flake_found)
                if self.save_all or flake_found:
                    status = "yes" if flake_found else "no"
                    stem   = f"{x}_{y}_{z}_{status}_{flake_size}"
                    path   = _unique_path(self.save_folder, stem)
                    cv2.imwrite(path, bgr)

                done += 1
                self.step_done.emit({
                    'done': done, 'total': total,
                    'slow_i': slow_i, 'fast_j': fast_j,
                    'x': x, 'y': y, 'z': z,
                    'flake_found': flake_found, 'flake_size': flake_size,
                })

            if self._stop:
                break

            # move slow axis (skip after the last row)
            if slow_i < self.slow_n - 1:
                if self.fast_axis == 'x':
                    mc.set_speed(self.slow_speed)
                    mc.move_y(self.slow_angle)
                else:
                    mc.set_speed(self.slow_speed)
                    mc.move_x(self.slow_angle)

        self.finished.emit()


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

        # defaults
        self.save_relevant_rad.setChecked(True)
        self.fast_x_rad.setChecked(True)

        # timer for live coord display
        self.coord_timer = QTimer()
        self.coord_timer.timeout.connect(self._update_coords)
        self.coord_timer.start(500)

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
        self.xpp.pressed.connect(lambda: self._start_continuous(self.mc().move_x,  CONTINUOUS_STEPS, self.x_speed_bx))
        self.xmm.pressed.connect(lambda: self._start_continuous(self.mc().move_x, -CONTINUOUS_STEPS, self.x_speed_bx))
        self.ypp.pressed.connect(lambda: self._start_continuous(self.mc().move_y,  CONTINUOUS_STEPS, self.y_speed_bx))
        self.ymm.pressed.connect(lambda: self._start_continuous(self.mc().move_y, -CONTINUOUS_STEPS, self.y_speed_bx))
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
        worker.finished.connect(lambda w=worker: self._workers.remove(w) if w in self._workers else None)
        self._current_worker = worker
        worker.start()

    def _on_move_done(self):
        self._worker_busy = False
        self.show_coords()
        self._dispatch()

    # ── continuous ──────────────────────────────────────────────────────────

    def _start_continuous(self, move_fn, steps, speed_bx):
        mc    = self.mc()
        speed = speed_bx.value()
        self._continuous_active = True
        self._continuous_fn = lambda: (mc.set_speed(speed), move_fn(steps))
        self._fire_continuous()

    def _fire_continuous(self):
        if not self._continuous_active:
            return
        worker = MotionWorker(self._continuous_fn)
        self._workers.append(worker)
        worker.done.connect(self._on_continuous_done)
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

    # ── single-step ─────────────────────────────────────────────────────────

    def xpf(self):
        mc = self.mc()
        speed, steps = self.x_speed_bx.value(), self.x_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_x(steps)))

    def xmf(self):
        mc = self.mc()
        speed, steps = self.x_speed_bx.value(), self.x_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_x(-steps)))

    def ypf(self):
        mc = self.mc()
        speed, steps = self.y_speed_bx.value(), self.y_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_y(steps)))

    def ymf(self):
        mc = self.mc()
        speed, steps = self.y_speed_bx.value(), self.y_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_y(-steps)))

    # ── multi-step ──────────────────────────────────────────────────────────

    def xpppf(self):
        mc = self.mc()
        speed, steps = self.x_speed_bx.value(), self.x_multible.value() * self.x_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_x(steps)))

    def xmmmf(self):
        mc = self.mc()
        speed, steps = self.x_speed_bx.value(), self.x_multible.value() * self.x_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_x(-steps)))

    def ypppf(self):
        mc = self.mc()
        speed, steps = self.y_speed_bx.value(), self.y_multible.value() * self.y_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_y(steps)))

    def ymmmf(self):
        mc = self.mc()
        speed, steps = self.y_speed_bx.value(), self.y_multible.value() * self.y_angle_bx.value()
        self._run(lambda: (mc.set_speed(speed), mc.move_y(-steps)))

    def _update_coords(self):
        mc = self.mc()
        if mc:
            self.coord_display.setText(f"X: {mc.get_x()}, Y: {mc.get_y()}, Z: {mc.get_z()}")

    def show_coords(self):
        self._update_coords()

    # ── scan ────────────────────────────────────────────────────────────────

    def start_scan(self):
        if self.mc() is None:
            print("[AutoScan] No motion controller — connect from the Manual tab first.")
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

        self.worker = ScanWorker(
            self.mc(), self.image_frame_manager, self.a_eye_tab.pipeline,
            fast_axis, fast_n, fast_angle, fast_speed,
            slow_n, slow_angle, slow_speed,
            save_all, save_folder,
            ratio, batch_size, radius,
        )
        self.worker.step_done.connect(self.on_step_done)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        print(f"[AutoScan] Started — {slow_n}×{fast_n} grid, fast={fast_axis}, save={'all' if save_all else 'detected'}")

    def stop_scan(self):
        if self.worker:
            self.worker.stop()
            print("[AutoScan] Stop requested.")

    def on_step_done(self, info):
        if 'error' in info:
            self.scan_info.setText(f"Error: {info['error']}")
            return
        x, y, z = info['x'], info['y'], info['z']
        self.coord_display.setText(f"X: {x}, Y: {y}, Z: {z}")
        flake_txt = f"flake=YES ({info['flake_size']}pts)" if info['flake_found'] else "flake=no"
        self.scan_info.setText(
            f"Step {info['done']}/{info['total']}  s={info['slow_i']} f={info['fast_j']}  {flake_txt}"
        )

    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.scan_info.setText("Scan finished.")
        print("[AutoScan] Scan finished.")
