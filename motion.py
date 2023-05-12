from copy import copy
import numpy as np
from operator import itemgetter
import pickle
import scipy.ndimage as sn
import time
from collections import defaultdict

from communication.server import  readmatrixfromjson,receive_data1,receive_data2,receive_data3, send_command1,send_command2,send_command3,command_joint,stop_threading,server_establish,stopserver
import matplotlib.pyplot as plt
import threading
from communication.client_socket_motion import motion_caption_establish, client_socket, set_target, takephoto_on, takephoto_off
from transform.transform import grid2local, local2world

#从txt中读取matrix
def readmatrixfromtxt(dir, num):
    A=np.zeros((num,num),dtype=float)
    f=open(dir)
    lines=f.readlines() #将全部数据读到一个lines中
    A_row=0         #表示矩阵的行，从0开始
    for line in lines:
        list=line.strip('\n').split(' ')
        A[A_row:]=list[0:num]
        A_row += 1
    return A

if __name__ == '__main__':
    PORT1 = 12340
    PORT2 = 12341
    PORT3 = 12342

    filename1 = './matrix1.json'
    filename2 = './matrix2.json'
    filename3 = './matrix3.json'
    
    IP_ADDRESS1 = '192.168.1.107'
    IP_ADDRESS2 = '192.168.1.107'
    IP_ADDRESS3 = '192.168.1.107'
    #双机运行
    #建立连接
    server_establish(IP_ADDRESS1,PORT1,IP_ADDRESS2,PORT2,IP_ADDRESS3,PORT3)

    # Create threads for sending and receiving data
    receive_thread1 = threading.Thread(target=receive_data1)
    receive_thread2 = threading.Thread(target=receive_data2)
    receive_thread3 = threading.Thread(target=receive_data3)

    # command_joint([1,1,1,1],0)
    #指令发送线程
    send_thread1 = threading.Thread(target=send_command1)
    send_thread2 = threading.Thread(target=send_command2)
    send_thread3 = threading.Thread(target=send_command3)
   
    # Start thread
    send_thread1.start()
    send_thread2.start()
    send_thread3.start()
    
    receive_thread1.start()     
    receive_thread2.start()     
    receive_thread3.start()
    
    motion_caption_establish()
    motion_thread = threading.Thread(target=client_socket)
    motion_thread.start()
    
    time.sleep(8)

    while True:
        label = 1 #or uavid 2, 3
        position = (0,0) #行，列
        position_local = grid2local(position)
        position_world = local2world(position_local)
        # print(position_world)
        set_target(label,position_world)
                
        # takephoto_on()
        # time.sleep(6)
        takephoto_off()
        time.sleep(2)
        #从文件中读取matrix
        observed_img1 = readmatrixfromjson(filename1)
        observed_img2 = readmatrixfromjson(filename2)
        observed_img3 = readmatrixfromjson(filename3)
            
            