import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

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


if __name__ == "__main__":
    unittest.main()
