import time
import stack_approach.Engine_Activation_XM as xm

class RollerGripper:
    LEFT_FINGER_PORT = 1
    LEFT_ROLLER_PORT = 2

    def __init__(self):
        self.portHandler, self.packetHandler = xm.open_port(self.LEFT_FINGER_PORT)

        xm.set_operating_mode([self.LEFT_ROLLER_PORT], "velocity_mode", self.portHandler, self.packetHandler)
        xm.set_operating_mode([self.LEFT_FINGER_PORT], "position_mode", self.portHandler, self.packetHandler)

        xm.set_torque([self.LEFT_ROLLER_PORT], True, self.portHandler, self.packetHandler)
        xm.set_torque([self.LEFT_FINGER_PORT], True, self.portHandler, self.packetHandler)

    def roll(self, wait_time=2):
        xm.vel_control([self.LEFT_ROLLER_PORT],[-80], self.portHandler, self.packetHandler)
        time.sleep(wait_time)
        xm.vel_control([self.LEFT_ROLLER_PORT],[0], self.portHandler, self.packetHandler)

    def close(self, final_pos=1000):
        xm.pos_control([self.LEFT_FINGER_PORT],[final_pos], self.portHandler, self.packetHandler)
        time.sleep(.5)
    
    def open(self):
        xm.pos_control([self.LEFT_FINGER_PORT],[2800], self.portHandler, self.packetHandler) 
        time.sleep(.5)

def main():
    roller = RollerGripper()
    roller.open()


if __name__ == '__main__':
    main()