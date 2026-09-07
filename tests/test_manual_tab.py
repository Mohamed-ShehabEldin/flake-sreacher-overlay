import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QSizePolicy

from flake_searcher.manual_tab import ManualTab
from flake_searcher.motion_controller import StageCapabilities


class CapabilityController:
    def __init__(self, axes):
        self.capabilities = StageCapabilities(frozenset(axes))

    def supports_axis(self, axis):
        return self.capabilities.supports_axis(axis)

    def is_connected(self):
        return True

    def is_position_valid(self):
        return True

    def get_x(self):
        return 0

    def get_y(self):
        return 0

    def get_z(self):
        return 0


class RecordingController(CapabilityController):
    def __init__(self):
        super().__init__({"X", "Y", "Z"})
        self.calls = []

    def move_x(self, steps, speed=None, owner=None):
        self.calls.append(("X", steps, speed, owner))

    def move_y(self, steps, speed=None, owner=None):
        self.calls.append(("Y", steps, speed, owner))

    def move_z(self, steps, speed=None, owner=None):
        self.calls.append(("Z", steps, speed, owner))


class ManualTabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_preconnection_movement_actions_fail_safely(self):
        tab = ManualTab()

        tab.xpf()
        tab.ypf()
        tab.move_to_x()
        tab.move_to_y()
        tab.xpp.pressed.emit()

        self.assertFalse(tab.has_active_motion())
        self.assertIn("Connect stage first", tab.MController_status.text())
        self.assertIn("not connected", tab.coord_display.text())

    def test_z_controls_are_disabled(self):
        tab = ManualTab()

        for widget in (
            tab.zp, tab.zm, tab.zpp, tab.zmm,
            tab.z_speed_bx, tab.z_angle_bx,
            tab.move_to_z_btn, tab.move_to_z_spinbx,
        ):
            self.assertFalse(widget.isEnabled())
            self.assertIn("firmware", widget.toolTip())
        self.assertTrue(tab.z_capability_note.isVisibleTo(tab))
        self.assertIn("X/Y only", tab.z_capability_note.text())

    def test_z_controls_enable_when_controller_reports_capability(self):
        tab = ManualTab()
        tab.motion_controller = CapabilityController({"X", "Y", "Z"})

        tab._refresh_axis_controls()
        tab.show_coords()

        for widget in (
            tab.zp, tab.zm, tab.zpp, tab.zmm,
            tab.z_speed_bx, tab.z_angle_bx,
            tab.move_to_z_btn, tab.move_to_z_spinbx,
        ):
            self.assertTrue(widget.isEnabled())
            self.assertEqual(widget.toolTip(), "")
        self.assertFalse(tab.z_capability_note.isVisibleTo(tab))
        self.assertIn("Z: 0", tab.coord_display.text())

    def test_scan_disables_jogging_but_keeps_safety_disconnect(self):
        tab = ManualTab()
        tab.motion_controller = CapabilityController({"X", "Y"})
        tab._refresh_connection_controls()

        tab.set_scan_active(True)

        self.assertFalse(tab.xp.isEnabled())
        self.assertFalse(tab.xpp.isEnabled())
        self.assertFalse(tab.move_to_x_btn.isEnabled())
        self.assertFalse(tab.push_connect_M.isEnabled())
        self.assertTrue(tab.disconnect_M_btn.isEnabled())

    def test_axis_button_order_labels_and_compact_size(self):
        tab = ManualTab()
        expected = (
            ("x_axis_layout", (("xm", 4, 0, "− Step"), ("xmm", 4, 1, "− Hold"),
                               ("xp", 5, 0, "+ Step"), ("xpp", 5, 1, "+ Hold"))),
            ("y_axis_layout", (("ym", 4, 0, "− Step"), ("ymm", 4, 1, "− Hold"),
                               ("yp", 5, 0, "+ Step"), ("ypp", 5, 1, "+ Hold"))),
            ("z_axis_layout", (("zm", 4, 0, "− Step"), ("zmm", 4, 1, "− Hold"),
                               ("zp", 5, 0, "+ Step"), ("zpp", 5, 1, "+ Hold"))),
        )

        for layout_name, buttons in expected:
            layout = getattr(tab, layout_name)
            for object_name, expected_row, expected_column, expected_text in buttons:
                with self.subTest(button=object_name):
                    button = getattr(tab, object_name)
                    index = layout.indexOf(button)
                    row, column, row_span, column_span = layout.getItemPosition(index)
                    self.assertEqual((row, column, row_span, column_span),
                                     (expected_row, expected_column, 1, 1))
                    self.assertEqual(button.objectName(), object_name)
                    self.assertEqual(button.text(), expected_text)
                    self.assertEqual(button.sizePolicy().verticalPolicy(), QSizePolicy.Fixed)
                    self.assertEqual(button.minimumHeight(), 32)
                    self.assertEqual(button.maximumHeight(), 32)

    def test_axis_button_signal_mappings_preserve_sign_and_action_type(self):
        tab = ManualTab()
        controller = RecordingController()
        tab.motion_controller = controller
        tab._refresh_axis_controls()
        tab._run = lambda operation: operation()

        for button_name in ("xm", "xp", "ym", "yp", "zm", "zp"):
            getattr(tab, button_name).pressed.emit()

        self.assertEqual(
            [(axis, steps) for axis, steps, _speed, _owner in controller.calls],
            [
                ("X", -1000.0), ("X", 1000.0),
                ("Y", -1000.0), ("Y", 1000.0),
                ("Z", -1000.0), ("Z", 1000.0),
            ],
        )

        controller.calls.clear()
        tab._fire_continuous = lambda: None
        for button_name in ("xmm", "xpp", "ymm", "ypp", "zmm", "zpp"):
            button = getattr(tab, button_name)
            button.pressed.emit()
            tab._continuous_fn()
            button.released.emit()

        self.assertEqual(
            [(axis, steps) for axis, steps, _speed, _owner in controller.calls],
            [
                ("X", -100), ("X", 100),
                ("Y", -100), ("Y", 100),
                ("Z", -100), ("Z", 100),
            ],
        )


if __name__ == "__main__":
    unittest.main()
