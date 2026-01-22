import sys
import termios
import tty
import select
import time
from collections import deque

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

import pyqtgraph as pg
from PyQt5 import QtWidgets


# =======================
# Control table addresses (XL430 / X-series)
# =======================
ADDR_MODEL_NUMBER = 0
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_CURRENT = 126
ADDR_GOAL_POSITION = 116

PROTOCOL_VERSION = 2.0
BAUDRATE = 57600
TICKS_PER_REV = 4096


# =======================
# Generic read / write helpers
# =======================
def dxl_read(packet, port, dxl_id, addr, size):
    if size == 1:
        value, result, error = packet.read1ByteTxRx(port, dxl_id, addr)
    elif size == 2:
        value, result, error = packet.read2ByteTxRx(port, dxl_id, addr)
    elif size == 4:
        value, result, error = packet.read4ByteTxRx(port, dxl_id, addr)
    else:
        raise ValueError("Invalid read size")

    if result != COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(result))
    if error != 0:
        raise RuntimeError(packet.getRxPacketError(error))

    return value


def dxl_write(packet, port, dxl_id, addr, value, size):
    if size == 1:
        result, error = packet.write1ByteTxRx(port, dxl_id, addr, value)
    elif size == 4:
        result, error = packet.write4ByteTxRx(port, dxl_id, addr, value)
    else:
        raise ValueError("Invalid write size")

    if result != COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(result))
    if error != 0:
        raise RuntimeError(packet.getRxPacketError(error))


# =======================
# Keyboard helpers
# =======================
def get_key():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


# =======================
# Live plotting class
# =======================
class LivePlotter:
    def __init__(self, min_limit, max_limit, window=300):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        self.window = window
        self.t_data = deque(maxlen=window)

        self.pos_data = deque(maxlen=window)
        self.goal_data = deque(maxlen=window)
        self.cur_data = deque(maxlen=window)

        self.start_time = time.time()

        self.win = pg.GraphicsLayoutWidget(title="Dynamixel Live Monitor")
        self.win.resize(900, 550)

        # ---- Position plot (ticks) ----
        self.pos_plot = self.win.addPlot(row=0, col=0, title="Position (ticks)")
        self.pos_curve = self.pos_plot.plot(pen=pg.mkPen("y", width=2), name="Actual")
        self.goal_curve = self.pos_plot.plot(
            pen=pg.mkPen("c", width=2, style=pg.QtCore.Qt.DashLine),
            name="Goal",
        )

        # Axis limits (+/- 5%)
        margin = int(0.05 * (max_limit - min_limit))
        self.pos_plot.setYRange(min_limit - margin, max_limit + margin)

        # ---- Current plot (mA) ----
        self.cur_plot = self.win.addPlot(row=1, col=0, title="Current (mA)")
        self.cur_curve = self.cur_plot.plot(pen=pg.mkPen("r", width=2))
        self.cur_plot.setYRange(-100, 1000)

        self.win.show()

        self.last_goal = None

    def update(self, pos_ticks, current_mA, goal_ticks=None):
        t = time.time() - self.start_time

        self.t_data.append(t)
        self.pos_data.append(pos_ticks)
        self.cur_data.append(current_mA)

        if self.last_goal is None:
            self.last_goal = pos_ticks

        # Keep last goal if none updated this cycle
        if goal_ticks is not None:
            self.last_goal = goal_ticks
        self.goal_data.append(self.last_goal)

        self.pos_curve.setData(self.t_data, self.pos_data)
        self.goal_curve.setData(self.t_data, self.goal_data)
        self.cur_curve.setData(self.t_data, self.cur_data)

        self.app.processEvents()


# =======================
# Main function
# =======================
def incremental_position_control(device_name, dxl_id, dq=1.0):
    port = PortHandler(device_name)
    packet = PacketHandler(PROTOCOL_VERSION)

    if not port.openPort():
        raise RuntimeError("Failed to open port")
    if not port.setBaudRate(BAUDRATE):
        raise RuntimeError("Failed to set baudrate")


    # ---- Motor info ----
    model = dxl_read(packet, port, dxl_id, ADDR_MODEL_NUMBER, 2)
    print(f"Connected to Dynamixel ID {dxl_id}, Model {model}")

    # ---- Read motor limits ----
    min_limit = dxl_read(packet, port, dxl_id, 52, 4)
    max_limit = dxl_read(packet, port, dxl_id, 48, 4)
    print(f"Motor position limits: {min_limit} – {max_limit}")

    plotter = LivePlotter(min_limit, max_limit)

    # ---- Enable torque ----
    if dxl_read(packet, port, dxl_id, ADDR_TORQUE_ENABLE, 1) == 0:
        dxl_write(packet, port, dxl_id, ADDR_TORQUE_ENABLE, 1, 1)

    print(f"Incremental control: Δq = {dq}°")
    print("↑ : +Δq | ↓ : −Δq | q : quit\n")

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)

    try:
        while True:
            key = get_key()

            # ---- Always update plot ----
            curr_ticks = dxl_read(packet, port, dxl_id, ADDR_PRESENT_POSITION, 4)
            curr_deg = curr_ticks * 360.0 / TICKS_PER_REV

            current_raw = dxl_read(packet, port, dxl_id, ADDR_PRESENT_CURRENT, 2)
            if current_raw > 32767:
                current_raw -= 65536
            current_mA = current_raw * 2.69

            plotter.update(
                pos_ticks=curr_ticks,
                current_mA=current_mA,
            )

            if not key:
                continue

            if key == "q":
                break

            if key == "\x1b":
                key += sys.stdin.read(2)
                if key not in ("\x1b[A", "\x1b[B"):
                    continue

                delta_deg = dq if key == "\x1b[A" else -dq
                target_deg = curr_deg + delta_deg
                target_ticks = int(target_deg * TICKS_PER_REV / 360.0)
                target_ticks = max(min_limit, min(max_limit, target_ticks))

                dxl_write(packet, port, dxl_id, ADDR_GOAL_POSITION, target_ticks, 4)
                
                plotter.update(
                    pos_ticks=curr_ticks,
                    current_mA=current_mA,
                    goal_ticks=target_ticks,
                )


                print(
                    f"pos={curr_deg:6.1f}°  "
                    f"cmd={target_ticks:5d}  "
                    f"I={current_mA:7.1f} mA"
                )

            time.sleep(0.02)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        port.closePort()
        print("Disconnected")


# =======================
# Example usage
# =======================
if __name__ == "__main__":
    incremental_position_control(
        device_name="/dev/dynamixel_left",
        dxl_id=4,
        dq=5.0,
    )
