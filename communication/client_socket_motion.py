import socket
import sys
import time
import numpy as np
sys.path.insert(0,'D:\distributed-sensing')
from transform.transform4 import transformHomogeneous
from pid_control.motion_capture_drone import pid_control
import threading
from communication.server import server_establish,receive_data1,send_command1,command_joint,receive_data2,send_command2,receive_data3,send_command3
from djitellopy import tello

take_photo = 0

target_loc = np.array(
    [[500, 1000, -1000], 
    [500, 1000, -400], 
    [500, 1000, 200]]
)

motion_socket = None

def takephoto_on():
    global take_photo
    take_photo = 1

def takephoto_off():
    global take_photo
    take_photo = 0
    
def set_target(uavid,target):
    global target_loc
    if uavid == 1:
        target_loc[0,0:2] = target
    elif uavid == 2:
        target_loc[1,0:2] = target
    elif uavid == 3:
        target_loc[2,0:2] = target
 
def motion_caption_establish():
    ################################################################
    # build connecction with VICON server
    ################################################################
    global motion_socket
    HOST, PORT = "127.0.0.1", 8888
    motion_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    motion_socket.connect((HOST, PORT))
    
def client_socket():
    global motion_socket
    while True:
        motion_socket.send(b'Hello')
        id_locations_str = motion_socket.recv(1000).decode()

        N_rigids = 3
        locations = np.zeros((N_rigids,3))
        for line in id_locations_str.splitlines():
            words = line.split()
            rigid_name = words[0]
            id = int(rigid_name[5]) # 'rigidX'
            locations[id-1] = np.genfromtxt(words[1:], dtype=float)

        global target_loc
        global take_photo
        for i in range(N_rigids):
            xVal = 0
            yVal = 0
            zVal = 0
            print(target_loc)
            xVal, yVal, zVal = pid_control(locations[i], target_loc[i])
            command_joint(i+1, [xVal, yVal, zVal,0], take_photo)

        time.sleep(0.1)


if __name__ == '__main__':
    pass