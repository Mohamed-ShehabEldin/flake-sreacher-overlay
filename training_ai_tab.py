from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5 import uic
import shutil
import os

from ai_logic import AutoScanPipeline

DEFAULT_HIERA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ai", "auto_scan_v1", "sam2.1_hiera_small.pt"
)


class PipelineWorker(QThread):
    done = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.fn()
            self.done.emit("done")
        except Exception as e:
            self.done.emit(f"error: {e}")


class TrainingAiTab(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("train_ai_tab.ui", self)

        self.pipeline = None
        self.worker = None

        # set default hiera path
        self.hierapath_lineEdit.setText(DEFAULT_HIERA)

        # tooltips
        self.savepath_lineEdit.setToolTip("Directory where datapoints/ folder and model.h5 will be saved.")
        self.hierapath_lineEdit.setToolTip("Path to the SAM2 checkpoint file (sam2.1_hiera_small.pt).\nUsed by 'collect invalid' to segment flakes.")
        self.max_display_width_spin.setToolTip("Downscale images wider than this for display during invalid data collection.\nDoes not affect the saved data quality.")
        self.grid_sample_size_spin.setToolTip("Number of sample points along each axis when building the invalid datapoints grid.\nHigher = more datapoints, slower collection.")
        self.epochs_spin.setToolTip("Maximum number of training epochs.\nEarly stopping will halt training sooner if the model stops improving.")
        self.batch_size_spin.setToolTip("Number of samples processed per gradient update.\nSmaller = slower but sometimes better generalisation. Default: 32.")
        self.test_size_spin.setToolTip("Fraction of data held out for final accuracy evaluation (e.g. 0.2 = 20%).\nNot used during training.")
        self.patience_spin.setToolTip("How many epochs to wait without improvement in val_loss before stopping early.\nPrevents overfitting.")

        self.savepath_btn.clicked.connect(self.pick_save_path)
        self.hierapath_btn.clicked.connect(self.pick_hiera_path)
        self.collect_valid_btn.clicked.connect(self.collect_valid)
        self.collect_invalid_btn.clicked.connect(self.collect_invalid)
        self.train_btn.clicked.connect(self.label_and_train)
        self.save_model_btn.clicked.connect(self.save_model)

    def pick_save_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select save directory")
        if folder:
            self.savepath_lineEdit.setText(folder)
            self.pipeline = AutoScanPipeline(save_dir=folder)
            print(f"[TrainTab] Save dir → {folder}")

    def pick_hiera_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SAM2 checkpoint", "", "Model (*.pt)")
        if path:
            self.hierapath_lineEdit.setText(path)

    def get_pipeline(self):
        if self.pipeline is None:
            save_dir = self.savepath_lineEdit.text().strip()
            if not save_dir:
                print("[TrainTab] Please select a save path first.")
                return None
            self.pipeline = AutoScanPipeline(save_dir=save_dir)
        return self.pipeline

    def get_hiera_path(self):
        return self.hierapath_lineEdit.text().strip() or DEFAULT_HIERA

    def run_in_thread(self, fn):
        self.worker = PipelineWorker(fn)
        self.worker.done.connect(lambda msg: print(f"[TrainTab] {msg}"))
        self.worker.start()

    def collect_valid(self):
        pipeline = self.get_pipeline()
        if not pipeline:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder with VALID flake images")
        if folder:
            pipeline.collect_valid(folder)  # must run on main thread (OpenCV GUI)

    def collect_invalid(self):
        pipeline = self.get_pipeline()
        if not pipeline:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder with INVALID flake images")
        if folder:
            pipeline.collect_invalid(
                folder,
                checkpoint=self.get_hiera_path(),
                max_display_width=self.max_display_width_spin.value(),
                grid_sample_size=self.grid_sample_size_spin.value(),
            )  # main thread

    def label_and_train(self):
        pipeline = self.get_pipeline()
        if pipeline:
            epochs     = self.epochs_spin.value()
            batch_size = self.batch_size_spin.value()
            test_size  = self.test_size_spin.value()
            patience   = self.patience_spin.value()
            def _run():
                pipeline.label()
                pipeline.train(epochs=epochs, batch_size=batch_size, test_size=test_size, patience=patience)
            self.run_in_thread(_run)

    def save_model(self):
        pipeline = self.get_pipeline()
        if not pipeline:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save model", "", "Model (*.h5)")
        if path:
            shutil.copy(str(pipeline.model_path), path)
            print(f"[TrainTab] Model saved → {path}")
