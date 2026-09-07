from PyQt5.QtWidgets import QWidget, QFileDialog, QDialog, QLabel, QScrollArea, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, QEvent, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5 import uic
import cv2
import os
import shutil

from .pipeline import AutoScanPipeline
from .image_frame_manager import ImageFrameManager
from .paths import ui_path
from .ui_feedback import set_status


class InferenceWorker(QThread):
    done = pyqtSignal(object, object, object)  # cls_mat, img_disp, stats
    error = pyqtSignal(str)

    def __init__(self, pipeline, image, ratio, batch_size, radius):
        super().__init__()
        self.pipeline   = pipeline
        self.image      = image   # file path or BGR numpy array
        self.ratio      = ratio
        self.batch_size = batch_size
        self.radius     = radius

    def run(self):
        try:
            cls_mat, img_disp, stats = self.pipeline.test(
                self.image,
                ratio=self.ratio,
                batch_size=self.batch_size,
                radius=self.radius,
            )
            self.done.emit(cls_mat, img_disp, stats)
        except Exception as e:
            self.error.emit(str(e))


class FolderInferenceWorker(QThread):
    done = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, pipeline, folder, ratio, batch_size, radius):
        super().__init__()
        self.pipeline   = pipeline
        self.folder     = folder
        self.ratio      = ratio
        self.batch_size = batch_size
        self.radius     = radius

    def run(self):
        try:
            output_folder = os.path.join(self.folder, "a_eye_results")
            os.makedirs(output_folder, exist_ok=True)

            total_files = 0
            saved_files = 0

            for name in os.listdir(self.folder):
                path = os.path.join(self.folder, name)
                if not os.path.isfile(path):
                    continue

                img = cv2.imread(path)
                if img is None:
                    continue

                total_files += 1

                _, _, stats = self.pipeline.test(
                    path,
                    ratio=self.ratio,
                    batch_size=self.batch_size,
                    radius=self.radius,
                )

                flake_size = stats['filtered']
                if flake_size <= 0:
                    continue

                new_name = f"s{flake_size:03d}_{name}"
                save_path = os.path.join(output_folder, new_name)
                shutil.copy2(path, save_path)
                saved_files += 1

            self.done.emit(
                f"Folder done. checked: {total_files}  saved: {saved_files}  folder: {output_folder}"
            )
        except Exception as e:
            self.error.emit(str(e))


class A_Eye_Tab(QWidget):
    def __init__(self, image_frame_manager: ImageFrameManager):
        super().__init__()
        uic.loadUi(str(ui_path("a_eye_tab.ui")), self)

        self.image_frame_manager = image_frame_manager
        self.pipeline = AutoScanPipeline()
        self.worker = None
        self._last_pixmap = None

        self.graphicsView.viewport().installEventFilter(self)

        self.ratio_spin.setToolTip("Grid density: one sample point every N pixels. Lower = more points, slower.")
        self.pred_batch_size_spin.setToolTip("Number of grid points sent to the model at once. Higher = faster (if GPU memory allows).")
        self.radius_spin.setToolTip("Radius in pixels of the circles drawn on detected flake locations.")

        self.chose_model_btn.clicked.connect(self.load_model)
        self.check_an_img_btn.clicked.connect(self.check_image)
        self.check_current_win_btn.clicked.connect(self.check_current_window)
        self.check_fldr_btn.clicked.connect(self.check_folder)

    def _set_status(self, text, state="neutral"):
        set_status(self.info, text, state)

    def _set_busy(self, busy):
        for button in (
            self.chose_model_btn,
            self.check_an_img_btn,
            self.check_current_win_btn,
            self.check_fldr_btn,
        ):
            button.setEnabled(not busy)

    def eventFilter(self, obj, event):
        if obj is self.graphicsView.viewport():
            if event.type() == QEvent.MouseButtonDblClick:
                self._open_fullsize()
            elif event.type() == QEvent.Resize:
                self._fit_result()
        return super().eventFilter(obj, event)

    def _fit_result(self):
        scene = self.graphicsView.scene()
        if scene is not None and not scene.itemsBoundingRect().isEmpty():
            self.graphicsView.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _open_fullsize(self):
        if self._last_pixmap is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("A-Eye Result — full size")
        dlg.resize(900, 700)
        label = QLabel()
        label.setPixmap(self._last_pixmap)
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(False)
        layout = QVBoxLayout()
        layout.addWidget(scroll)
        dlg.setLayout(layout)
        dlg.show()

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select model", "", "Model (*.h5)")
        if path:
            self.chose_model_lineEdit.setText(path)
            try:
                self.pipeline.load_model_from_path(path)
                self._set_status(f"Model loaded: {os.path.basename(path)}", "connected")
                print(f"[A-Eye] Model loaded ← {path}")
            except Exception as e:
                self._set_status(f"Model load failed — {e}", "error")
                print(f"[A-Eye] Error loading model: {e}")

    def _get_params(self):
        return (
            self.ratio_spin.value(),
            self.pred_batch_size_spin.value(),
            self.radius_spin.value(),
        )

    def _run_inference(self, image):
        if self.pipeline._model is None:
            self._set_status("Choose a detector model first.", "warning")
            print("[A-Eye] Please load a model first.")
            return
        ratio, batch_size, radius = self._get_params()
        self.worker = InferenceWorker(self.pipeline, image, ratio, batch_size, radius)
        self._set_busy(True)
        self._set_status("Running inference…", "busy")
        self.worker.done.connect(self._show_result)
        self.worker.error.connect(self._on_inference_error)
        self.worker.start()

    def check_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.tif *.bmp)")
        if path:
            self._run_inference(path)

    def check_folder(self):
        if self.pipeline._model is None:
            self._set_status("Choose a detector model first.", "warning")
            print("[A-Eye] Please load a model first.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if not folder:
            return

        ratio, batch_size, radius = self._get_params()
        self.worker = FolderInferenceWorker(self.pipeline, folder, ratio, batch_size, radius)
        self._set_busy(True)
        self._set_status("Checking folder…", "busy")
        self.worker.done.connect(self._on_folder_done)
        self.worker.error.connect(self._on_folder_error)
        self.worker.start()

    def check_current_window(self):
        try:
            rgb = self.image_frame_manager.get_screenshot()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self._run_inference(bgr)
        except Exception as e:
            self._set_status(f"Microscope capture failed — {e}", "error")
            print(f"[A-Eye] Screenshot failed: {e}")

    def _show_result(self, _cls_mat, img_disp, stats):
        h, w = img_disp.shape[:2]
        rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self._last_pixmap = pixmap

        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.graphicsView.setScene(scene)
        self._fit_result()

        info_text = (
            f"flakes: {stats['filtered']}  (raw: {stats['raw']})  |  "
            f"{stats['h']}x{stats['w']}px  |  "
            f"bg: {stats['bg']}  |  "
            f"{stats['elapsed']:.1f}s"
        )
        self._set_busy(False)
        self._set_status(info_text, "connected")
        print(f"[A-Eye] {info_text}")

    def _on_inference_error(self, message):
        self._set_busy(False)
        self._set_status(f"Inference failed — {message}", "error")
        print(f"[A-Eye] Error: {message}")

    def _on_folder_done(self, msg):
        self._set_busy(False)
        self._set_status(msg, "connected")
        print(f"[A-Eye] {msg}")

    def _on_folder_error(self, msg):
        self._set_busy(False)
        self._set_status(f"Folder check failed — {msg}", "error")
        print(f"[A-Eye] Error: {msg}")
