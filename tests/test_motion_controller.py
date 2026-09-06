import threading
import time
import unittest
from collections import deque

import serial

from flake_searcher.motion_controller import (
    InvalidMotionCommandError,
    MotionBusyError,
    MotionController,
    MotionNotConnectedError,
    StageCapabilities,
    StageCommunicationError,
    UnsupportedAxisError,
)


def speed_ack(delay=195):
    rps = 1_000_000 // (2 * delay * 6400)
    return f"Speed set: stepDelay={delay}us (~{rps} rev/s)\n".encode()


class FakeSerial:
    def __init__(self, responses=()):
        self.responses = deque(responses)
        self.writes = []
        self.timeout = None
        self.is_open = True

    def write(self, payload):
        self.writes.append(payload.decode().strip())
        return len(payload)

    def flush(self):
        pass

    def readline(self):
        if not self.responses:
            return b""
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response

    def reset_input_buffer(self):
        pass

    def close(self):
        self.is_open = False


class DynamicSerial(FakeSerial):
    def __init__(self):
        super().__init__()
        self._responses = deque()
        self._lock = threading.Lock()

    def write(self, payload):
        command = payload.decode().strip()
        with self._lock:
            self.writes.append(command)
            if command.startswith("S "):
                delay = int(command.split()[1])
                self._responses.append(speed_ack(delay))
            else:
                axis, steps = command.split()
                self._responses.append(f"{axis} moved {steps} steps.\n".encode())
        return len(payload)

    def readline(self):
        time.sleep(0.005)
        with self._lock:
            return self._responses.popleft()


def controller_with(fake, capabilities=None):
    return MotionController(
        "FAKE",
        serial_factory=lambda *args, **kwargs: fake,
        boot_wait_ms=0,
        capabilities=capabilities,
    )


class MotionControllerTests(unittest.TestCase):
    def test_correct_acknowledgement_changes_coordinate(self):
        fake = FakeSerial([speed_ack(), b"X moved 100 steps.\n"])
        controller = controller_with(fake)

        result = controller.move_x(100, speed=0.4)

        self.assertEqual(result, 100)
        self.assertEqual(controller.get_x(), 100)
        self.assertTrue(controller.is_position_valid())
        self.assertEqual(fake.writes, ["S 195", "X 100"])

    def test_incorrect_acknowledgement_invalidates_position(self):
        fake = FakeSerial([speed_ack(), b"Y moved 100 steps.\n"])
        controller = controller_with(fake)

        with self.assertRaises(StageCommunicationError):
            controller.move_x(100, speed=0.4)

        self.assertEqual(controller.get_x(), 0)
        self.assertFalse(controller.is_position_valid())
        self.assertFalse(controller.is_connected())

    def test_malformed_response_invalidates_position(self):
        fake = FakeSerial([speed_ack(), b"\xff\xfe\n"])
        controller = controller_with(fake)

        with self.assertRaises(StageCommunicationError):
            controller.move_x(100, speed=0.4)

        self.assertFalse(controller.is_position_valid())

    def test_timeout_is_finite_and_invalidates_position(self):
        fake = FakeSerial([speed_ack(), b""])
        controller = controller_with(fake)

        with self.assertRaisesRegex(StageCommunicationError, "Timed out"):
            controller.move_x(100, speed=0.4)

        self.assertGreaterEqual(fake.timeout, 2.0)
        self.assertFalse(controller.is_position_valid())

    def test_serial_disconnect_invalidates_position(self):
        fake = FakeSerial([speed_ack(), serial.SerialException("disconnected")])
        controller = controller_with(fake)

        with self.assertRaises(StageCommunicationError):
            controller.move_x(100, speed=0.4)

        self.assertFalse(controller.is_position_valid())
        self.assertFalse(controller.is_connected())

    def test_closed_serial_is_not_reported_as_valid_position(self):
        fake = FakeSerial()
        controller = controller_with(fake)
        fake.is_open = False

        self.assertFalse(controller.is_connected())
        self.assertFalse(controller.is_position_valid())
        with self.assertRaises(MotionNotConnectedError):
            controller.move_x(100, speed=0.4)

    def test_coordinate_only_changes_after_successful_ack(self):
        fake = FakeSerial([
            speed_ack(), b"X moved 100 steps.\n",
            speed_ack(), b"invalid\n",
        ])
        controller = controller_with(fake)
        controller.move_x(100, speed=0.4)

        with self.assertRaises(StageCommunicationError):
            controller.move_x(50, speed=0.4)

        self.assertEqual(controller.get_x(), 100)
        self.assertFalse(controller.is_position_valid())

    def test_z_is_unsupported_and_never_written(self):
        fake = FakeSerial()
        controller = controller_with(fake)

        with self.assertRaises(UnsupportedAxisError):
            controller.move_z(100, speed=0.4)

        self.assertEqual(fake.writes, [])
        self.assertEqual(controller.get_z(), 0)

    def test_z_follows_controller_capability(self):
        fake = FakeSerial([speed_ack(), b"Z moved 100 steps.\n"])
        capabilities = StageCapabilities(frozenset({"X", "Y", "Z"}))
        controller = controller_with(fake, capabilities=capabilities)

        controller.move_z(100, speed=0.4)

        self.assertEqual(fake.writes, ["S 195", "Z 100"])
        self.assertEqual(controller.get_z(), 100)

    def test_fractional_steps_are_rejected_without_writing(self):
        fake = FakeSerial()
        controller = controller_with(fake)

        with self.assertRaises(InvalidMotionCommandError):
            controller.move_x(0.5, speed=0.4)

        self.assertEqual(fake.writes, [])
        self.assertTrue(controller.is_position_valid())

    def test_invalid_speed_is_rejected_without_writing(self):
        fake = FakeSerial()
        controller = controller_with(fake)

        for speed in (None, 0, -1):
            with self.subTest(speed=speed):
                with self.assertRaises(InvalidMotionCommandError):
                    controller.move_x(100, speed=speed)

        self.assertEqual(fake.writes, [])
        self.assertTrue(controller.is_position_valid())

    def test_preconnection_motion_fails_without_writing(self):
        def failed_factory(*args, **kwargs):
            raise serial.SerialException("not present")

        controller = MotionController("MISSING", serial_factory=failed_factory, boot_wait_ms=0)

        with self.assertRaises(MotionNotConnectedError):
            controller.move_x(100, speed=0.4)

        self.assertFalse(controller.is_position_valid())

    def test_concurrent_transactions_do_not_interleave(self):
        fake = DynamicSerial()
        controller = controller_with(fake)
        barrier = threading.Barrier(3)
        errors = []

        def move(fn):
            try:
                barrier.wait()
                fn(100, speed=0.4)
            except Exception as e:
                errors.append(e)

        x_thread = threading.Thread(target=move, args=(controller.move_x,))
        y_thread = threading.Thread(target=move, args=(controller.move_y,))
        x_thread.start()
        y_thread.start()
        barrier.wait()
        x_thread.join()
        y_thread.join()

        self.assertEqual(errors, [])
        self.assertIn(
            fake.writes,
            (["S 195", "X 100", "S 195", "Y 100"],
             ["S 195", "Y 100", "S 195", "X 100"]),
        )
        self.assertEqual((controller.get_x(), controller.get_y()), (100, 100))

    def test_exclusive_scan_owner_rejects_competing_motion(self):
        fake = FakeSerial([speed_ack(), b"X moved 100 steps.\n"])
        controller = controller_with(fake)
        scan_owner = object()
        self.assertTrue(controller.acquire_exclusive(scan_owner))

        with self.assertRaises(MotionBusyError):
            controller.move_y(100, speed=0.4, owner=object())

        controller.move_x(100, speed=0.4, owner=scan_owner)
        self.assertEqual(fake.writes, ["S 195", "X 100"])

    def test_reconnect_closes_previous_controller(self):
        first = FakeSerial()
        second = FakeSerial()
        serials = iter((first, second))
        controller = MotionController(
            "FAKE",
            serial_factory=lambda *args, **kwargs: next(serials),
            boot_wait_ms=0,
        )

        self.assertTrue(controller.connect_device())

        self.assertFalse(first.is_open)
        self.assertTrue(second.is_open)
        self.assertTrue(controller.is_position_valid())


if __name__ == "__main__":
    unittest.main()
