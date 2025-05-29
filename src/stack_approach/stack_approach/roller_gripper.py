import time
import stack_approach.Engine_Activation_XM as xm

class RollerGripper:

    def __init__(self):
        self.portHandler, self.packetHandler = xm.open_port(3)
        xm.set_operating_mode([3], "velocity_mode", self.portHandler, self.packetHandler)
        xm.set_torque([3], True, self.portHandler, self.packetHandler)

    def roll(self, wait_time=2):
        xm.vel_control([3],[-80], self.portHandler, self.packetHandler)
        time.sleep(wait_time)
        xm.vel_control([3],[0], self.portHandler, self.packetHandler)

def main():
    roller = RollerGripper()
    roller.roll()


if __name__ == '__main__':
    main()