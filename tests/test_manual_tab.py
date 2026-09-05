import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from manual_tab import ManualTab


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

    def test_scan_disables_jogging_but_keeps_safety_disconnect(self):
        tab = ManualTab()

        tab.set_scan_active(True)

        self.assertFalse(tab.xp.isEnabled())
        self.assertFalse(tab.xpp.isEnabled())
        self.assertFalse(tab.move_to_x_btn.isEnabled())
        self.assertTrue(tab.push_connect_M.isEnabled())
        self.assertEqual(tab.push_connect_M.text(), "Disconnect")


if __name__ == "__main__":
    unittest.main()
