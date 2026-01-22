from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

DEVICENAME = "/dev/dynamixel_left"
BAUDRATE = 57600
DXL_ID = 3

port = PortHandler(DEVICENAME)
packet = PacketHandler(2.0)

port.openPort()
port.setBaudRate(BAUDRATE)

ADDR_MODEL_NUMBER = 0   # Model Number address
ADDR_PRESENT_POSITION = 132   # XL430 Present Position (4 bytes)

model_number, dxl_result, dxl_error = packet.read2ByteTxRx(port, DXL_ID, ADDR_MODEL_NUMBER)

if dxl_result == COMM_SUCCESS and dxl_error == 0:
    print("Model Number:", model_number)


position, dxl_result, dxl_error = packet.read4ByteTxRx(
    port, DXL_ID, ADDR_PRESENT_POSITION
)

if dxl_result == COMM_SUCCESS and dxl_error == 0:
    print("Current position (raw ticks):", position)

port.closePort()

