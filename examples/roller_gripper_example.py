import Engine_Activation_XM as xm
import time

finger_port = 1
roller_port = 2


portHandler, packetHandler = xm.open_port(finger_port)

xm.set_operating_mode([finger_port], "position_mode", portHandler, packetHandler)
xm.set_operating_mode([roller_port], "velocity_mode", portHandler, packetHandler)

xm.set_torque([finger_port, roller_port], True, portHandler, packetHandler)

while True:
    command = input("give command").strip().lower()

    if command == "c":
        xm.vel_control([roller_port],[-80], portHandler, packetHandler) # positive: away from the fingers
        time.sleep(5)
        xm.vel_control([roller_port],[0], portHandler, packetHandler)

        xm.pos_control([finger_port],[0], portHandler, packetHandler) # Close 
    elif command == "o":
        xm.pos_control([finger_port],[2200], portHandler, packetHandler) # Open
    elif command == "q":
        print("bye")
        break
    else:
        print("unknown command", command)