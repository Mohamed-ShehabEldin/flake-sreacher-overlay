import unittest
from unittest.mock import patch

import numpy as np

from flake_searcher.autoscan_tab import ScanWorker
from flake_searcher.motion_controller import StageCommunicationError


class FakeStage:
    def __init__(self):
        self.x = 0
        self.y = 0

    def is_connected(self):
        return True

    def is_position_valid(self):
        return True

    def move_x(self, steps, speed=None, owner=None):
        self.x += int(steps)

    def move_y(self, steps, speed=None, owner=None):
        self.y += int(steps)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return 0


class RecordingFrameManager:
    def __init__(self, stage):
        self.stage = stage
        self.positions = []

    def get_screenshot(self):
        self.positions.append((self.stage.get_x(), self.stage.get_y()))
        return np.zeros((10, 10, 3), dtype=np.uint8)


class EmptyPipeline:
    _model = None


class FailingStage(FakeStage):
    def __init__(self):
        super().__init__()
        self.valid = True

    def is_position_valid(self):
        return self.valid

    def move_x(self, steps, speed=None, owner=None):
        self.valid = False
        raise StageCommunicationError("lost acknowledgement")


def run_scan(zigzag):
    stage = FakeStage()
    frames = RecordingFrameManager(stage)
    worker = ScanWorker(
        stage, frames, EmptyPipeline(),
        "x", 3, 10, 0.4,
        2, 20, 0.4,
        False, "", 5, 100, 1, zigzag, object(),
    )
    with patch("flake_searcher.autoscan_tab.time.sleep", return_value=None):
        worker.run()
    return frames.positions


class ScanSequenceTests(unittest.TestCase):
    def test_normal_scan_coordinate_sequence_is_unchanged(self):
        self.assertEqual(
            run_scan(zigzag=False),
            [(0, 0), (10, 0), (20, 0), (0, 20), (10, 20), (20, 20)],
        )

    def test_zigzag_scan_coordinate_sequence_is_unchanged(self):
        self.assertEqual(
            run_scan(zigzag=True),
            [(0, 0), (10, 0), (20, 0), (20, 20), (10, 20), (0, 20)],
        )

    def test_stage_failure_aborts_scan_and_emits_error(self):
        stage = FailingStage()
        frames = RecordingFrameManager(stage)
        worker = ScanWorker(
            stage, frames, EmptyPipeline(),
            "x", 3, 10, 0.4,
            2, 20, 0.4,
            False, "", 5, 100, 1, False, object(),
        )
        errors = []
        worker.stage_failed.connect(errors.append)

        with patch("flake_searcher.autoscan_tab.time.sleep", return_value=None):
            worker.run()

        self.assertEqual(frames.positions, [(0, 0)])
        self.assertEqual(errors, ["lost acknowledgement"])


if __name__ == "__main__":
    unittest.main()
