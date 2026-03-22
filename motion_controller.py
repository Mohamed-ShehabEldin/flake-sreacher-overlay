from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5 import QtTest
import serial


class MotionWorker(QThread):
    """Run a motion function in a background thread so the GUI stays responsive."""
    done = pyqtSignal()

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()
        self.done.emit()

class MotionController:
    def __init__(self, com_port):
        self.com_port = com_port
        self.motion_controller = None  # Will be initialized upon connection
        self.absolute_x = 0
        self.absolute_y = 0
        self.absolute_z = 0
        self.ser = None
        self.connect_device()

    def connect_device(self):
        try:
            # timeout=2 only during boot so reset_input_buffer doesn't hang
            self.ser = serial.Serial(self.com_port, baudrate=2000000, timeout=2)
            QtTest.QTest.qWait(2000)       # wait for Arduino to finish resetting
            self.ser.reset_input_buffer()  # discard the "Ready." boot message
            self.ser.timeout = None        # now block until each move ACK arrives
            return True
        except Exception as e:
            print(e)
            return False
        
    def move_x(self, step=1, speed=100):
        '''Move the X axis (NEMA 17 via TB6600).
        step: Number of steps (positive = forward, negative = backward).'''
        if self.ser:
            command = f'X {step}\n'.encode()
            self.ser.write(command)
            self.ser.readline()  # wait for Arduino acknowledgment
            self.absolute_x += step

    def move_y(self, step=1, speed=100):
        '''Move the Y axis (NEMA 17 via TB6600).
        step: Number of steps (positive = forward, negative = backward).'''
        if self.ser:
            command = f'Y {step}\n'.encode()
            self.ser.write(command)
            self.ser.readline()  # wait for Arduino acknowledgment
            self.absolute_y += step

    def move_z(self, step=1, speed=100):
        '''Move the Z axis (small stepper, TBD).
        step: Number of steps (positive = forward, negative = backward).'''
        if self.ser:
            command = f'Z {step}\n'.encode()
            self.ser.write(command)
            self.ser.readline()  # wait for Arduino acknowledgment
            self.absolute_z += step

    def set_speed(self, revs_per_sec):
        '''Set motor speed. Converts rev/sec to Arduino stepDelay (µs).
        revs_per_sec: desired speed (e.g. 1.0 = 1 rev/sec, max ~5 rev/sec).'''
        if self.ser and revs_per_sec > 0:
            step_delay = int(1_000_000 / (2 * revs_per_sec * 6400))
            command = f'S {step_delay}\n'.encode()
            self.ser.write(command)
            self.ser.readline()

    def get_x(self):
        return self.absolute_x
    def get_y(self):
        return self.absolute_y
    def get_z(self):
        return self.absolute_z

    def disconnect(self):
        if self.ser:
            self.ser.close()
            self.ser = None


if __name__ == "__main__":
    print("__main__")
    # ── Test configuration ──────────────────────────────────────────
    COM_PORT   = "/dev/cu.usbserial-1140"   # change to your Arduino port (e.g. "/dev/ttyUSB0" on Linux/Mac)
    SPEED      = 1      # rev/sec
    STEPS      = 9000      # steps to move in each direction
    REPEATS    = 2        # how many back-and-forth cycles to run
    # ────────────────────────────────────────────────────────────────

    import time

    mc = MotionController(COM_PORT)
    mc.set_speed(SPEED)

    print(f"Starting test: {REPEATS} cycles, {STEPS} steps each side, {SPEED} rev/s")

    for i in range(REPEATS):
        print(f"Cycle {i+1}/{REPEATS} — moving X+ Y+")
        mc.move_x(STEPS)
        mc.move_y(STEPS)

        print(f"Cycle {i+1}/{REPEATS} — moving X- Y-")
        mc.move_x(-STEPS)
        mc.move_y(-STEPS)

    mc.disconnect()
    print("Test complete.")