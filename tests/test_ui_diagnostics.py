import unittest

from flake_searcher.ui_diagnostics import summarize_heartbeats


class UiDiagnosticsTests(unittest.TestCase):
    def test_heartbeat_summary_reports_phase_and_excess_latency(self):
        summary = summarize_heartbeats(
            [
                {"phase": "baseline", "interval_ms": 10.0},
                {"phase": "baseline", "interval_ms": 12.0},
                {"phase": "background_inference", "interval_ms": 35.0},
            ],
            target_interval_ms=10.0,
        )
        self.assertEqual(summary["overall"]["max_ms"], 35.0)
        self.assertEqual(summary["overall"]["max_excess_ms"], 25.0)
        self.assertEqual(summary["by_phase"]["background_inference"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
