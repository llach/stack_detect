import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

from stack_msgs.srv import RollerGripper, RollerGripperV2

# Finger / roller ports
CAMERA_FINGER_PORT = 1
CAMERA_ROLLER_PORT = 2
NORMAL_ROLLER_PORT = 3
NORMAL_FINGER_PORT = 4

# Dynamixel addresses
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_GOAL_VELOCITY = 104
ADDR_MIN_POS_LIMIT = 52
ADDR_MAX_POS_LIMIT = 48
ADDR_PRESENT_CURRENT = 126

PROTOCOL_VERSION = 2.0
BAUDRATE = 57600
TICKS_PER_REV = 4096

# operating mode constants
POSITION_MODE = 3
VELOCITY_MODE = 1

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


# Helper functions
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
    elif size == 2:
        result, error = packet.write2ByteTxRx(port, dxl_id, addr, value)
    elif size == 4:
        result, error = packet.write4ByteTxRx(port, dxl_id, addr, value)
    else:
        raise ValueError("Invalid write size")

    if result != COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(result))
    if error != 0:
        raise RuntimeError(packet.getRxPacketError(error))

def read_current_mA(portHandler, packetHandler, FINGER_PORT):
    raw = dxl_read(packetHandler, portHandler, FINGER_PORT, ADDR_PRESENT_CURRENT, 2)
    if raw > 32767:
        raw -= 65536
    return raw * 2.69  # XL430 scale

def set_torque(portHandler, packetHandler, dxl_ids, enable=True):
    for dxl_id in dxl_ids:
        dxl_write(packetHandler, portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE if enable else TORQUE_DISABLE, 1)

def set_operating_mode(portHandler, packetHandler, dxl_ids, mode):
    for dxl_id in dxl_ids:
        # Must disable torque before changing mode
        dxl_write(packetHandler, portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, 1)
        dxl_write(packetHandler, portHandler, dxl_id, ADDR_OPERATING_MODE, mode, 1)

def move_position(portHandler, packetHandler, dxl_id, goal_position):
    dxl_write(packetHandler, portHandler, dxl_id, ADDR_GOAL_POSITION, goal_position, 4)


def move_velocity(portHandler, packetHandler, dxl_id, velocity):
    dxl_write(packetHandler, portHandler, dxl_id, ADDR_GOAL_VELOCITY, velocity, 4)


# =====================================
# ROS2 Service Node
# =====================================
class RollerGripperService(Node):
    def __init__(self):
        super().__init__('roller_gripper')

        # ROS parameters
        self.declare_parameter("finger_port", 1)
        self.declare_parameter("roller_port", 2)

        self.FINGER_PORT = self.get_parameter('finger_port').get_parameter_value().integer_value
        self.ROLLER_PORT = self.get_parameter('roller_port').get_parameter_value().integer_value

        assert self.FINGER_PORT in [CAMERA_FINGER_PORT, NORMAL_FINGER_PORT], f"unknown finger port {self.FINGER_PORT}"
        assert self.ROLLER_PORT in [CAMERA_ROLLER_PORT, NORMAL_ROLLER_PORT], f"unknown roller port {self.ROLLER_PORT}"

        self.SRV_PREFIX = "right" if self.FINGER_PORT == CAMERA_FINGER_PORT else "left"
        # Direction: right = +1, left = -1 (motors mirrored)
        self.DIR = 1 if self.SRV_PREFIX == "right" else -1
        self.get_logger().info(f"Finger direction multiplier DIR = {self.DIR}")


        # -------------------
        # Setup Dynamixel ports
        # -------------------
        self.portHandler = PortHandler(f"/dev/dynamixel_{self.SRV_PREFIX}")
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        if not self.portHandler.openPort():
            raise RuntimeError("Failed to open port")
        if not self.portHandler.setBaudRate(BAUDRATE):
            raise RuntimeError("Failed to set baudrate")

        # Set operating modes
        set_operating_mode(self.portHandler, self.packetHandler, [self.FINGER_PORT], POSITION_MODE)
        set_operating_mode(self.portHandler, self.packetHandler, [self.ROLLER_PORT], VELOCITY_MODE)

        # Enable torque
        set_torque(self.portHandler, self.packetHandler, [self.FINGER_PORT, self.ROLLER_PORT], True)

        # Read motor limits
        self.min_pos_limit = dxl_read(self.packetHandler, self.portHandler, self.FINGER_PORT, ADDR_MIN_POS_LIMIT, 4)
        self.max_pos_limit = dxl_read(self.packetHandler, self.portHandler, self.FINGER_PORT, ADDR_MAX_POS_LIMIT, 4)

        self.get_logger().info(f"Finger limits: {self.min_pos_limit} – {self.max_pos_limit}")

        self.srv = self.create_service(
            RollerGripper, 
            f'{self.SRV_PREFIX}_roller_gripper', 
            self.srv_callback
            )
        self.norm_srv = self.create_service(
            RollerGripperV2,
            f'{self.SRV_PREFIX}_gripper_normalized',
            self.normalized_callback
        )

        self.get_logger().info("Dynamixel setup done!")
        self.calibrate()

    def destroy_node(self):
        self.shutdown_motors()
        super().destroy_node()

    def calibrate(self, step=10, freq=20, current_limit=250):
        """
        step: ticks per move
        freq: Hz command rate
        current_limit: mA threshold to detect contact
        """
        self.get_logger().info("Starting finger calibration...")

        period = 1.0 / freq

        # ---- Move to HALF interval position first (ASSUMPTION: ) ----
        frist_pos = (self.max_pos_limit+self.min_pos_limit) // 2
        self.get_logger().info(f"Moving to FIRST position {frist_pos}")
        move_position(self.portHandler, self.packetHandler, self.FINGER_PORT, frist_pos)

        # ---- Determine closing direction ----
        # For right side motor, closing means decreasing ticks
        # For left side motor, closing means increasing ticks
        step_signed = -step * self.DIR

        pos = frist_pos

        self.get_logger().info("Closing slowly until contact...")

        while True:
            pos += step_signed

            # Clamp to motor range
            pos = max(self.min_pos_limit, min(self.max_pos_limit, pos))
            move_position(self.portHandler, self.packetHandler, self.FINGER_PORT, pos)

            time.sleep(period)

            current = read_current_mA(self.portHandler, self.packetHandler, self.FINGER_PORT)

            # self.get_logger().info(f"pos={pos} current={current:.1f} mA")

            if abs(current) > current_limit:
                self.get_logger().info("Contact detected!")
                break

        self.calibrated_closed_pos = pos

        self.get_logger().info(f"Calibration complete. Closed position = {pos}")
        self.print_calibrated_pose()

        self.setup_normalized_range()
        move_position(self.portHandler, self.packetHandler, self.FINGER_PORT, self.command_to_ticks(1.0))

    def setup_normalized_range(self, open_span=1300, close_span=150):
        """
        Defines normalized motion around the calibrated contact point.

        +1.0 → open direction (away from object)
        -1.0 → deeper close (into object, force zone)
        """

        contact = self.calibrated_closed_pos

        # Direction that moves AWAY from object (opening)
        open_sign = self.DIR          # mirrored motors handled here
        close_sign = -self.DIR        # direction into object

        desired_open_limit = contact + open_sign * open_span
        desired_close_limit = contact + close_sign * close_span

        # Clamp to mechanical limits
        self.open_limit = max(self.min_pos_limit, min(self.max_pos_limit, desired_open_limit))
        self.close_limit = max(self.min_pos_limit, min(self.max_pos_limit, desired_close_limit))

        if self.open_limit != desired_open_limit:
            self.get_logger().warn("!!! Open span clipped by motor limits !!!")

        if self.close_limit != desired_close_limit:
            self.get_logger().warn("!!! Close span clipped by motor limits !!!")

        self.contact_pos = contact  # store for mapping

        self.get_logger().info(
            f"\nNormalized control configured around contact:\n"
            f"  Close limit (-1.0): {self.close_limit}\n"
            f"  Contact    ( 0.0): {self.contact_pos}\n"
            f"  Open limit  (+1.0): {self.open_limit}\n"
        )

    def command_to_ticks(self, cmd):
        cmd = max(-1.0, min(1.0, cmd))

        if cmd >= 0.0:
            # 0 → +1  : contact → open
            span = self.open_limit - self.contact_pos
            target = self.contact_pos + cmd * span
        else:
            # 0 → -1  : contact → deeper close
            span = self.contact_pos - self.close_limit
            target = self.contact_pos + cmd * span  # cmd negative

        return int(target)

    def print_calibrated_pose(self, bar_width = 50):
        def to_bar(p):
            return int((p - self.min_pos_limit) / (self.max_pos_limit - self.min_pos_limit) * bar_width)

        closed_i = to_bar(self.calibrated_closed_pos)

        # Determine which side is open
        open_is_max = (self.DIR == 1)

        bar = ["-"] * (bar_width + 1)

        if open_is_max:
            bar[0] = "|"            # closed-side mechanical limit
            bar[bar_width] = "O"    # open mechanical limit
        else:
            bar[0] = "O"            # open mechanical limit
            bar[bar_width] = "|"    # closed-side mechanical limit

        bar[closed_i] = "C"

        bar_str = "".join(bar)

        self.get_logger().info(
            "\n"
            "🎯 Finger calibration complete\n"
            f"Mechanical range: [{self.min_pos_limit} … {self.max_pos_limit}] ticks\n"
            f"{self.min_pos_limit:<6d} {bar_str} {self.max_pos_limit:>6d}\n"
            f"{' ' * (7 + closed_i)}C = Contact ({self.calibrated_closed_pos} ticks)\n"
            f"{' ' * (7 + (bar_width if open_is_max else 0))}O = Open (mechanical limit)\n"
        )

    def normalized_callback(self, req, res):
        try:
            cmd = req.position
            ticks = self.command_to_ticks(cmd)

            self.get_logger().info(f"Normalized cmd {cmd:.2f} → ticks {ticks}")

            move_position(self.portHandler, self.packetHandler, self.FINGER_PORT, ticks)
            res.success = True

        except Exception as e:
            self.get_logger().error(f"Normalized command failed: {e}")
            res.success = False

        return res

    def shutdown_motors(self):
        self.get_logger().info("Disabling torque and closing port...")
        try:
            set_torque(self.portHandler, self.packetHandler,
                    [self.FINGER_PORT, self.ROLLER_PORT], False)
            self.portHandler.closePort()
        except Exception as e:
            self.get_logger().warn(f"Error during shutdown: {e}")

    def srv_callback(self, req, res):
        self.get_logger().info(f"Received gripper request: {req}")

        try:
            # ------------------------
            # Move finger
            # ------------------------
            if req.finger_pos != -1:
                self.get_logger().info(f"Moving finger to {req.finger_pos}")

                # Validate ranges
                if self.FINGER_PORT == CAMERA_FINGER_PORT:
                    assert 700 <= req.finger_pos <= 2500
                elif self.FINGER_PORT == NORMAL_FINGER_PORT:
                    assert 2000 <= req.finger_pos <= 3500

                move_position(self.portHandler, self.packetHandler, self.FINGER_PORT, req.finger_pos)
                time.sleep(0.5)

            # ------------------------
            # Roll roller
            # ------------------------
            elif req.roller_duration != -1:
                self.get_logger().info(f"Rolling at vel {req.roller_vel} for {req.roller_duration}")

                assert -100 <= req.roller_vel <= 100
                move_velocity(self.portHandler, self.packetHandler, self.ROLLER_PORT, req.roller_vel)
                time.sleep(req.roller_duration)
                move_velocity(self.portHandler, self.packetHandler, self.ROLLER_PORT, 0)

            else:
                self.get_logger().warn("Malformed request")
                res.success = False
                return res

        except Exception as e:
            self.get_logger().error(f"Error handling request: {e}")
            res.success = False
            return res

        self.get_logger().info("Request completed successfully")
        res.success = True
        return res


# =====================================
# ROS2 main
# =====================================
def main():
    rclpy.init()
    node = RollerGripperService()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Keyboard interrupt")

if __name__ == '__main__':
    main()
