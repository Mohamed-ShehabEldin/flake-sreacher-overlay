import math
import numbers
import threading

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5 import QtTest
import serial


STEPS_PER_REV = 6400
DEFAULT_STEP_DELAY_US = 78
MIN_STEP_DELAY_US = 10
MAX_STEP_DELAY_US = 10000
MIN_ACK_TIMEOUT_S = 2.0
ACK_MARGIN_S = 2.0


class MotionError(RuntimeError):
    pass


class MotionNotConnectedError(MotionError):
    pass


class MotionPositionUnknownError(MotionError):
    pass


class MotionBusyError(MotionError):
    pass


class InvalidMotionCommandError(MotionError):
    pass


class UnsupportedAxisError(MotionError):
    pass


class StageCommunicationError(MotionError):
    pass


class MotionWorker(QThread):
    """Run a motion function in a background thread so the GUI stays responsive."""
    done = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.fn()
        except Exception as e:
            print(f"[MotionWorker] {e}")
            self.failed.emit(str(e))
        finally:
            self.done.emit()


class MotionController:
    def __init__(self, com_port, serial_factory=None, boot_wait_ms=2000):
        self.com_port = com_port
        self.motion_controller = None
        self.absolute_x = 0
        self.absolute_y = 0
        self.absolute_z = 0
        self.ser = None
        self.position_valid = False
        self.position_error = "Not connected"
        self._step_delay_us = DEFAULT_STEP_DELAY_US
        self._serial_factory = serial_factory or serial.Serial
        self._boot_wait_ms = boot_wait_ms
        self._transaction_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._exclusive_owner = None
        self.connect_device()

    def connect_device(self):
        with self._transaction_lock:
            self.disconnect()
            new_serial = None
            try:
                new_serial = self._serial_factory(
                    self.com_port,
                    baudrate=2000000,
                    timeout=2,
                    write_timeout=2,
                )
                with self._state_lock:
                    self.ser = new_serial
                QtTest.QTest.qWait(self._boot_wait_ms)
                new_serial.reset_input_buffer()
                new_serial.timeout = MIN_ACK_TIMEOUT_S
                with self._state_lock:
                    self.ser = new_serial
                    self.absolute_x = 0
                    self.absolute_y = 0
                    self.absolute_z = 0
                    self._step_delay_us = DEFAULT_STEP_DELAY_US
                    self.position_valid = True
                    self.position_error = ""
                return True
            except Exception as e:
                self._invalidate_and_close(f"Connection failed: {e}")
                print(e)
                return False

    def is_connected(self):
        with self._state_lock:
            ser = self.ser
            return ser is not None and getattr(ser, "is_open", True)

    def is_position_valid(self):
        with self._state_lock:
            ser = self.ser
            connected = ser is not None and getattr(ser, "is_open", True)
            return self.position_valid and connected

    def acquire_exclusive(self, owner):
        if owner is None:
            raise ValueError("An owner token is required.")
        if not self._transaction_lock.acquire(blocking=False):
            return False
        try:
            if not self.is_connected() or not self.is_position_valid():
                return False
            with self._state_lock:
                if self._exclusive_owner not in (None, owner):
                    return False
                self._exclusive_owner = owner
            return True
        finally:
            self._transaction_lock.release()

    def release_exclusive(self, owner):
        with self._state_lock:
            if self._exclusive_owner is owner:
                self._exclusive_owner = None

    def _validate_owner_locked(self, owner):
        with self._state_lock:
            exclusive_owner = self._exclusive_owner
        if exclusive_owner is not None and exclusive_owner is not owner:
            raise MotionBusyError("Stage is reserved by the active automatic scan.")

    @staticmethod
    def _validate_steps(step):
        if isinstance(step, bool) or not isinstance(step, numbers.Real):
            raise InvalidMotionCommandError("Motor steps must be an integer.")
        value = float(step)
        if not math.isfinite(value) or not value.is_integer():
            raise InvalidMotionCommandError("Fractional motor steps are not supported.")
        value = int(value)
        if not -2_147_483_647 <= value <= 2_147_483_647:
            raise InvalidMotionCommandError("Motor step command is outside the firmware range.")
        return value

    @staticmethod
    def _speed_to_delay(speed):
        if isinstance(speed, bool) or not isinstance(speed, numbers.Real):
            raise InvalidMotionCommandError("Motor speed must be a positive number.")
        speed = float(speed)
        if not math.isfinite(speed) or speed <= 0:
            raise InvalidMotionCommandError("Motor speed must be greater than zero.")
        delay = int(1_000_000 / (2 * speed * STEPS_PER_REV))
        if not MIN_STEP_DELAY_US <= delay <= MAX_STEP_DELAY_US:
            min_speed = 1_000_000 / (2 * MAX_STEP_DELAY_US * STEPS_PER_REV)
            max_speed = 1_000_000 / (2 * MIN_STEP_DELAY_US * STEPS_PER_REV)
            raise InvalidMotionCommandError(
                f"Motor speed must be between {min_speed:.6f} and {max_speed:.4f} rev/s."
            )
        return delay

    def _require_ready_locked(self):
        if not self.is_connected():
            self._invalidate_and_close("Stage controller disconnected; position is unknown.")
            raise MotionNotConnectedError("Stage controller is not connected.")
        if not self.is_position_valid():
            raise MotionPositionUnknownError(
                self.position_error or "Stage position is unknown; reconnect before moving."
            )

    def _read_ack_locked(self, expected, timeout):
        ser = self.ser
        ser.timeout = timeout
        raw = ser.readline()
        if not raw:
            raise StageCommunicationError(
                f"Timed out after {timeout:.1f}s waiting for: {expected}"
            )
        try:
            response = raw.decode("utf-8").strip()
        except UnicodeDecodeError as e:
            raise StageCommunicationError("Stage returned a malformed response.") from e
        if response != expected:
            raise StageCommunicationError(
                f"Unexpected stage response: {response!r}; expected {expected!r}."
            )

    def _write_command_locked(self, command, expected, timeout):
        ser = self.ser
        payload = f"{command}\n".encode("ascii")
        written = ser.write(payload)
        if written is not None and written != len(payload):
            raise StageCommunicationError("Stage serial write was incomplete.")
        if hasattr(ser, "flush"):
            ser.flush()
        self._read_ack_locked(expected, timeout)

    def _set_speed_locked(self, speed):
        delay = self._speed_to_delay(speed)
        expected_rps = 1_000_000 // (2 * delay * STEPS_PER_REV)
        expected = f"Speed set: stepDelay={delay}us (~{expected_rps} rev/s)"
        self._write_command_locked(f"S {delay}", expected, MIN_ACK_TIMEOUT_S)
        self._step_delay_us = delay
        return delay

    def set_speed(self, revs_per_sec, owner=None):
        self._speed_to_delay(revs_per_sec)
        with self._transaction_lock:
            self._validate_owner_locked(owner)
            self._require_ready_locked()
            try:
                return self._set_speed_locked(revs_per_sec)
            except (InvalidMotionCommandError, MotionBusyError):
                raise
            except Exception as e:
                self._communication_failed(e)

    def _move(self, axis, step, speed=None, owner=None):
        if axis == "Z":
            raise UnsupportedAxisError("Z axis is unavailable in the current firmware.")
        steps = self._validate_steps(step)
        self._speed_to_delay(speed)

        with self._transaction_lock:
            self._validate_owner_locked(owner)
            self._require_ready_locked()
            try:
                self._set_speed_locked(speed)
                expected_motion_s = abs(steps) * 2 * self._step_delay_us / 1_000_000
                timeout = max(
                    MIN_ACK_TIMEOUT_S,
                    expected_motion_s * 1.5 + ACK_MARGIN_S,
                )
                expected = f"{axis} moved {steps} steps."
                self._write_command_locked(f"{axis} {steps}", expected, timeout)
            except (InvalidMotionCommandError, MotionBusyError):
                raise
            except Exception as e:
                self._communication_failed(e)

            with self._state_lock:
                if axis == "X":
                    self.absolute_x += steps
                    return self.absolute_x
                self.absolute_y += steps
                return self.absolute_y

    def move_x(self, step=1, speed=None, owner=None):
        return self._move("X", step, speed=speed, owner=owner)

    def move_y(self, step=1, speed=None, owner=None):
        return self._move("Y", step, speed=speed, owner=owner)

    def move_z(self, step=1, speed=None, owner=None):
        raise UnsupportedAxisError("Z axis is unavailable in the current firmware.")

    def _communication_failed(self, error):
        if isinstance(error, StageCommunicationError):
            message = str(error)
        else:
            message = f"Serial communication failed: {error}"
        print(f"[MC] {message}")
        self._invalidate_and_close(message)
        raise StageCommunicationError(message) from error

    def _invalidate_and_close(self, reason):
        with self._state_lock:
            ser = self.ser
            self.ser = None
            self.position_valid = False
            self.position_error = reason
            self._exclusive_owner = None
        if ser:
            try:
                ser.close()
            except Exception:
                pass

    def get_x(self):
        with self._state_lock:
            return self.absolute_x

    def get_y(self):
        with self._state_lock:
            return self.absolute_y

    def get_z(self):
        return 0

    def disconnect(self):
        self._invalidate_and_close("Disconnected; stage position is unknown.")


if __name__ == "__main__":
    print("__main__")
    COM_PORT = "/dev/cu.usbserial-1140"
    SPEED = 1
    STEPS = 9000
    REPEATS = 2

    mc = MotionController(COM_PORT)
    for i in range(REPEATS):
        print(f"Cycle {i+1}/{REPEATS} — moving X+ Y+")
        mc.move_x(STEPS, speed=SPEED)
        mc.move_y(STEPS, speed=SPEED)
        print(f"Cycle {i+1}/{REPEATS} — moving X- Y-")
        mc.move_x(-STEPS, speed=SPEED)
        mc.move_y(-STEPS, speed=SPEED)
    mc.disconnect()
    print("Test complete.")
