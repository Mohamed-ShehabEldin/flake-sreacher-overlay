import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PyQt5.QtWidgets import QApplication

from flake_searcher.a_eye_tab import A_Eye_Tab
from flake_searcher.autoscan_tab import AutoScan
from flake_searcher.manual_tab import ManualTab
from flake_searcher.paths import PROJECT_ROOT, UI_ROOT, ui_path
from flake_searcher.training_tab import TrainingAiTab


class _FrameManager:
    pass


class ApplicationPackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_ui_paths_are_absolute_and_exist(self):
        self.assertTrue(PROJECT_ROOT.is_absolute())
        self.assertTrue(UI_ROOT.is_absolute())
        for name in (
            "main_window.ui",
            "manual_tab.ui",
            "autoscan_tab.ui",
            "a_eye_tab.ui",
            "train_ai_tab.ui",
        ):
            self.assertTrue(ui_path(name).is_file())

    def test_tabs_load_when_working_directory_is_elsewhere(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            os.chdir(temporary_directory)
            try:
                manual = ManualTab()
                training = TrainingAiTab()
                a_eye = A_Eye_Tab(_FrameManager())
                auto = AutoScan(manual, _FrameManager(), a_eye)
                for widget in (auto, a_eye, training, manual):
                    widget.close()
            finally:
                os.chdir(previous)

    def test_application_import_does_not_load_training_frameworks(self):
        script = """
import builtins
blocked = ('tensorflow', 'torch', 'torchvision', 'sam2', 'sklearn')
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in blocked or name.startswith(tuple(item + '.' for item in blocked)):
        raise AssertionError('heavy dependency imported during startup: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
import flake_searcher.main_window
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        environment["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tempfile.gettempdir(),
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
