import os

# if os.name == 'nt':
#     import msvcrt
#     def getch():
#         return msvcrt.getch().decode()
# else:
#     import sys, tty, termios
#     fd = sys.stdin.fileno()
#     old_settings = termios.tcgetattr(fd)
#     def getch():
#         try:
#             tty.setraw(sys.stdin.fileno())
#             ch = sys.stdin.read(1)
#         finally:
#             termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#         return ch

from dynamixel_sdk import * # Uses Dynamixel SDK library
import serial.tools.list_ports
import time

# Control Table Adress
BAUDRATE                = 57600
PROTOCOL_VERSION        = 2.0
TORQUE_ENABLE           = 1
ADDR_CURRENT_LIMIT      = 38
ADDR_TORQUE_ENABLE      = 64
ADDR_GOAL_VELOCITY      = 104
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_VELOCITY   = 128
ADDR_PRESENT_POSITION   = 132
ADDR_OPERATING_MODE     = 11
LEN_GOAL_VELOCITY       = 4                     # Data Byte Length
LEN_PRESENT_VELOCITY    = 4                     # Data Byte Length
LEN_TORQUE_ENABLE       = 1
LEN_OPERATING_MODE      = 1
LEN_CURRENT_LIMIT       = 2
LEN_GOAL_POSITION       = 4


# Funció per detectar el port automàticament
def detect_dynamixel_port(DXL_ID):
    
    ports = serial.tools.list_ports.comports()
    for port in ports:
        portHandler = PortHandler(port.device)
        try:
            # Intenta obrir el port
            if portHandler.openPort():
                if portHandler.setBaudRate(BAUDRATE):
                    packetHandler = PacketHandler(PROTOCOL_VERSION)
                    # Intenta activar el torque per comprovar la connexió
                    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
                    if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                        print(f"Port detectat per motor ID{DXL_ID}: {port.device}")
                        portHandler.closePort()  # Tanquem el port perquè el codi principal el pugui utilitzar
                        return port.device  # Retorna el nom del port correcte
                portHandler.closePort()  # Tanca el port si no és el correcte
        except:
            print("skipping", port.device)
    print(f"No s'ha trobat cap port correcte per al motor amb ID{DXL_ID}")
    return None  # Si no troba cap port correcte, retorna None

def open_port(DXL_ID):

    PORT = detect_dynamixel_port(DXL_ID)
    portHandler = PortHandler(PORT)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open port
    if not portHandler.openPort():
        print("Failed to open port")
        quit()

    # Set port baudrate
    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to change the baudrate")
    print(f"Port: {PORT} obert")
    return portHandler, packetHandler

def close_port(portHandler):

    # Close port
    portHandler.closePort()
    print("Port Closed")

def set_torque(motor_ids, enable, portHandler, packetHandler):
    
    """
    Funció per activar o desactivar el torque dels motos
    - motor_ids: Llista dels IDs dels motors
    - enable: "True" per activar el torque i "False" per desactivar-lo
    """
    if enable == True:
        TORQUE_ENABLE = 1
    else:
        TORQUE_ENABLE = 0

    # Crear Sync Write
    groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_TORQUE_ENABLE, LEN_TORQUE_ENABLE)

    # Afegir les dades per a cada motor
    for motor_id in motor_ids:
        param_torque_enable = [TORQUE_ENABLE]
        dxl_addparam_result = groupSyncWrite.addParam(motor_id, param_torque_enable)
        if not dxl_addparam_result:
            print(f"Error afegint ID: {motor_id} a Sync Write")
            return
        
    # Enviar el paquet de dades    
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Error de comunicació: {packetHandler.getTxRxResult(dxl_comm_result)}")
    else:
        if enable:
            print("Torque activat per a tots els motors")
        else:
            print("Torque desactivat per a tots els motors")
    
    # Esborrar els paràmetres de Sync Write
    groupSyncWrite.clearParam()

def set_operating_mode(motor_ids, mode_name, portHandler, packetHandler):

    """
    mode_name: 
        - "current_mode": Current Control Mode (0)
        - "velocity_mode": Velocity Control Mode (1)
        - "position_mode": Position Control Mode (3)
        - "extended_position": Extended Position Control Mode (Multi-turn) (4)
        - "current_based": Current-based Position Control Mode (5)
        - "voltage_mode": PWM Control Mode (Voltage Control Mode) (16)
    """
    set_torque(motor_ids, False, portHandler, packetHandler)

    MODES = {"current_mode":0,"velocity_mode":1,"position_mode":3, "extended_position":4, "current_based":5, "voltage_mode":16}
    if mode_name not in MODES:
        print(f"Error: {mode_name} no és un mode vàlid. Modes disponibles: {list(MODES.keys())}")
        return
    operating_mode = MODES[mode_name]

    groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_OPERATING_MODE, LEN_OPERATING_MODE)

    # Afegir les dades per a cada motor
    for motor_id in motor_ids:
        param_operating_mode = [operating_mode]
        dxl_addparam_result = groupSyncWrite.addParam(motor_id, param_operating_mode)
        if not dxl_addparam_result:
            print(f"Error afegint ID: {motor_id} a Sync Write")
            return

    # Enviar el paquet de dades    
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Error de comunicació: {packetHandler.getTxRxResult(dxl_comm_result)}")
    else:
        print(f"Mode d'operació modificat a '{mode_name}' per als motors: {motor_ids}.")
    
    # Esborrar els paràmetres de Sync Write
    groupSyncWrite.clearParam()

def current_pos(motor_ids, current_limits, goal_positions, portHandler, packetHandler):
    """
    Funció per configurar el límit de corrent i la posició objectiu per a varis motors.
    - motor_ids: Llista d'IDs dels motors
    - current_limits: Llista de límits de corrent per a cada motor
    - goal_positions: Llista de les posicions per a cada motor
    """

    # Crear Sync Write
    groupSyncWrite1 = GroupSyncWrite(portHandler, packetHandler, ADDR_CURRENT_LIMIT, LEN_CURRENT_LIMIT)
    groupSyncWrite2 = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION) 

    ## SET CURRENT LIMIT ##
    # Afegir les dades per a cada motor
    for motor_id, current_limit in zip(motor_ids, current_limits):
        param_current_limit = [
            current_limit & 0xFF,
            (current_limit >> 8) & 0xFF
        ]
        dxl_addparam_result = groupSyncWrite1.addParam(motor_id, param_current_limit)
        if not dxl_addparam_result:
            print(f"Error afegint ID: {motor_id} a Sync Write")
            return
        
    # Enviar el paquet de dades
    dxl_comm_result = groupSyncWrite1.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Error de comunicació: {packetHandler.getTxRxResult(dxl_comm_result)}")
    else:
        print(f"Current Limit modificat a {current_limit}, al motor {motor_id}")

    ## WRITING GOAL POSITION ##
    # Afegir les dades per a cada motor
    for motor_id, goal_position in zip(motor_ids, goal_positions):
        param_goal_position = [
            goal_position & 0xFF,
            (goal_position >> 8) & 0xFF,
            (goal_position >> 16) & 0xFF,
            (goal_position >> 24) & 0xFF
        ]
        dxl_addparam_result = groupSyncWrite2.addParam(motor_id, param_goal_position)
    if not dxl_addparam_result:
        print(f"Error afegint ID: {motor_id} a Sync Write")
        return
    
    # Enviar el paquet de dades
    dxl_comm_result = groupSyncWrite2.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Error de comunicació: {packetHandler.getTxRxResult(dxl_comm_result)}")
    
    # Esborrar els paràmetres de Sync Write
    groupSyncWrite1.clearParam()
    groupSyncWrite2.clearParam()

def pos_control(motor_ids, goal_positions, portHandler, packetHandler):
    """
    Funció per controlar la posició objectiu de varis motors.
    - motor_ids: Llista d'IDs dels motors
    - goal_positions: Llista de les posicions per a cada motor, les posicions han de ser nombres enters
    """

    # Crear Sync Write
    groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION) 

    # Afegir les dades per a cada motor
    for motor_id, goal_position in zip(motor_ids, goal_positions):
        param_goal_position = [
            goal_position & 0xFF,
            (goal_position >> 8) & 0xFF,
            (goal_position >> 16) & 0xFF,
            (goal_position >> 24) & 0xFF
        ]
        dxl_addparam_result = groupSyncWrite.addParam(motor_id, param_goal_position)
    if not dxl_addparam_result:
        print(f"Error afegint ID: {motor_id} a Sync Write")
        return
    
    # Enviar el paquet de dades
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Error de comunicació: {packetHandler.getTxRxResult(dxl_comm_result)}")
    
    # Esborrar els paràmetres de Sync Write
    groupSyncWrite.clearParam()

def vel_control(motor_ids, goal_velocities, portHandler, packetHandler):
    """
    Funció per controlar la velocitat de x números de motors dynamixel
    Tant els IDs dels motors com les velocitats desitjades han de ser introduïdes com una llista
    És necessàri que els motors tinguin l'operating mode - "velocity_mode": Velocity Control Mode (1)
    - motor_ids: Llista d'IDs dels motors
    - goal_velocities: Llista de les velocitats per a cada motor
    """
    if len(motor_ids) != len(goal_velocities):
        print("Error: El nombre d'IDs i velocitats no coincideix")
        return

    groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_VELOCITY, LEN_GOAL_VELOCITY)

    # Afegir les dades per a cada motor
    for motor_id, goal_velocity in zip(motor_ids, goal_velocities):
        param_goal_velocity = [
            goal_velocity & 0xFF,
            (goal_velocity >> 8) & 0xFF,
            (goal_velocity >> 16) & 0xFF,
            (goal_velocity >> 24) & 0xFF
        ]

        dxl_addparam_result = groupSyncWrite.addParam(motor_id, param_goal_velocity)
        if not dxl_addparam_result:
            print(f"Error afegint ID:{motor_id} a Sync Write")
            return

    # Enviar el paquet de dades
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Error de comunicació: {packetHandler.getTxRxResult(dxl_comm_result)}")
    else:
        print("Comanda enviada correctament a tots el motors")
    
    # Esborrar els paràmetres de Sync Write
    groupSyncWrite.clearParam()

def read_present_velocity(DXL_ID, portHandler, packetHandler):
    """
    Funció per llegir la velocitat actual del motor especificat.
    Retorna la velocitat actual del motor en unitats del motor.
    """

    # Read Present Velocity
    dxl_present_velocity, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_VELOCITY)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Error en llegir la velocitat per al motor ID{DXL_ID}: {packetHandler.getTxRxResult(dxl_comm_result)}")
        return None
    if dxl_error != 0:
        print(f"Error de paquet")
        return None

    # Tractament de valors negatius
    if dxl_present_velocity > 2**31 - 1:
        dxl_present_velocity -= 2**32
    # print(f"Velocitat acutal del motor ID{DXL_ID}: {dxl_present_velocity}")
    time.sleep(0.01)
    return dxl_present_velocity

def read_present_position(DXL_ID, portHandler, packetHandler):
    """
    Funció per llegir la posició actual del motor especificat
    """

    # Read Present Position
    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Error en llegir la posició per al motor ID{DXL_ID}: {packetHandler.getTxRxResult(dxl_comm_result)}")
        return None
    if dxl_error != 0:
        print(f"Error de paquet")
        return None
    
    # print(f"Posició actual del motor ID{DXL_ID}: {dxl_present_position}")
    time.sleep(0.01)
    return dxl_present_position