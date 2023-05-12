import threading
import time
import sys
import json
import socket
import numpy as np

#三套参数，全局变量，文件内使用
filename1 = './matrix1.json'
server_socket1 = ''
client_socket1 = ''
client_address1 = ''
COMMAND1 = [0,0,0,0,0]
stop_sending = False
stop_receiving = False

filename2 = './matrix2.json'
server_socket2 = ''
client_socket2 = ''
client_address2 = ''
COMMAND2 = [0,0,0,0,0]

filename3 = './matrix3.json'
server_socket3 = ''
client_socket3 = ''
client_address3 = ''
COMMAND3 = [0,0,0,0,0]

mutex1 = threading.Lock()
mutex2 = threading.Lock()
mutex3 = threading.Lock()

#接收文件的进程
def receive_data1():
    global mutex1
    global stop_receiving
    global client_socket1
    global server_socket1
    global filename1
    while not stop_receiving:
        # Receive data from socket
        while True:
            received_data = client_socket1.recv(1024)
            if not received_data:
                break
            # Deserialize data from JSON and write to file
            decoded_data = received_data.decode()
            print(f"從客戶端 {client_address1} 收到訊息：{decoded_data}")
            stop = decoded_data.find('STOPRECEIVED')    
            if stop != -1:
                break
            n1 = decoded_data.find('[[')    
            n2 = 0 
            if n1 != -1:
                n2 = decoded_data.find(']]',n1)           
            if n1 != -1 and n2 != -1:
                decoded_data = decoded_data[n1:n2+2]
                print(1,decoded_data)
                data = json.loads(decoded_data)
                mutex1.acquire()
                with open(filename1, 'w') as f:
                    json.dump(data, f)
                mutex1.release()

def receive_data2():
    global mutex2
    global stop_receiving
    global client_socket2
    global server_socket2
    global filename2
    while not stop_receiving:
        # Receive data from socket
        while True:
            received_data = client_socket2.recv(1024)
            if not received_data:
                break
            # Deserialize data from JSON and write to file
            decoded_data = received_data.decode()
            print(f"從客戶端 {client_address2} 收到訊息：{decoded_data}")
            stop = decoded_data.find('STOPRECEIVED')    
            if stop != -1:
                break
            n1 = decoded_data.find('[[')    
            n2 = 0 
            if n1 != -1:
                n2 = decoded_data.find(']]',n1)           
            if n1 != -1 and n2 != -1:
                decoded_data = decoded_data[n1:n2+2]
                print(2,decoded_data)
                data = json.loads(decoded_data)
                mutex2.acquire()
                with open(filename2, 'w') as f:
                    json.dump(data, f)
                mutex2.release()

def receive_data3():
    global mutex3
    global stop_receiving
    global client_socket3
    global server_socket3
    global filename3
    while not stop_receiving:
        # Receive data from socket
        while True:

            received_data = client_socket3.recv(1024)
            # if len(received_data)<1024:
            #     received_data = client_socket3.recv(len(received_data))
            if not received_data:
                break
            # Deserialize data from JSON and write to file
            decoded_data = received_data.decode()
            print(f"從客戶端 {client_address3} 收到訊息：{decoded_data}")
            stop = decoded_data.find('STOPRECEIVED')    
            if stop != -1:
                break
            n1 = decoded_data.find('[[')    
            n2 = 0 
            if n1 != -1:
                n2 = decoded_data.find(']]',n1)           
            if n1 != -1 and n2 != -1:
                decoded_data = decoded_data[n1:n2+2]
                print(3,decoded_data)
                data = json.loads(decoded_data)
                mutex3.acquire()
                with open(filename3, 'w') as f:
                    json.dump(data, f)
                mutex3.release()

#发送指令的进程
def send_command1():
    global COMMAND1
    STOPCOMMAND = 'STOP'
    global stop_sending
    global client_socket1
    global server_socket1
    while not stop_sending:
        client_socket1.send(str(COMMAND1).encode())
        # print(1,COMMAND1)
        time.sleep(0.2)
    #发送结束指令
    client_socket1.send(str(STOPCOMMAND).encode())

def send_command2():
    global COMMAND2
    STOPCOMMAND = 'STOP'
    global stop_sending
    global client_socket2
    global server_socket2
    while not stop_sending:
        client_socket2.send(str(COMMAND2).encode())
        # print(2,COMMAND2)
        time.sleep(0.2)
    #发送结束指令
    client_socket2.send(str(STOPCOMMAND).encode())

def send_command3():
    global COMMAND3
    STOPCOMMAND = 'STOP'
    global stop_sending
    global client_socket3
    global server_socket3
    while not stop_sending:
        client_socket3.send(str(COMMAND3).encode())
        # print(3,COMMAND3)
        time.sleep(0.2)
    #发送结束指令
    client_socket3.send(str(STOPCOMMAND).encode())

#指令拼接函数
def command_joint(uavid,uav_control, colorrecognize_flag):
    global COMMAND1
    global COMMAND2
    global COMMAND3
    if uavid == 1:
        COMMAND1[0:4] = uav_control
        COMMAND1[4] = colorrecognize_flag
    elif uavid == 2:
        COMMAND2[0:4] = uav_control
        COMMAND2[4] = colorrecognize_flag
    elif uavid == 3:
        COMMAND3[0:4] = uav_control
        COMMAND3[4] = colorrecognize_flag

#停止线程
def stop_threading():
    global stop_sending
    global stop_receiving
    stop_sending = True
    stop_receiving = True

#服务端，建立监听过程
def server_establish(IP1,PORT1,IP2,PORT2,IP3,PORT3):
# def server_establish(IP1,PORT1):
    #三组链接
    global server_socket1
    global client_socket1
    global client_address1

    global server_socket2
    global client_socket2
    global client_address2

    global server_socket3
    global client_socket3
    global client_address3
    # 建立 Socket 物件
    server_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 綁定 IP 和端口
    server_socket1.bind((IP1, PORT1))
    server_socket2.bind((IP2, PORT2))
    server_socket3.bind((IP3, PORT3))
    # 監聽客戶端連線
    server_socket1.listen()
    print(f"伺服器正在監聽 {IP1}:{PORT1}")
    server_socket2.listen()
    print(f"伺服器正在監聽 {IP2}:{PORT2}")
    server_socket3.listen()
    print(f"伺服器正在監聽 {IP3}:{PORT3}")
    # 接受客戶端連線
    client_socket1, client_address1 = server_socket1.accept()
    print(f"客戶端 {client_address1} 已連線")
    client_socket2, client_address2 = server_socket2.accept()
    print(f"客戶端 {client_address2} 已連線")
    client_socket3, client_address3 = server_socket3.accept()
    print(f"客戶端 {client_address3} 已連線")

def stopserver():
    global server_socket1
    global server_socket2
    global server_socket3

    server_socket1.close()
    server_socket2.close()
    server_socket3.close()

    #从json文件中读取传入的matrix
def readmatrixfromjson(dir):
    global mutex1
    global mutex2
    global mutex3
    if dir == filename1:
        mutex1.acquire()
    elif dir == filename2:
        mutex2.acquire()
    elif dir == filename3:
        mutex3.acquire()
    with open(dir, 'r') as f:
        matrix_list = json.load(f)
    
    # Convert the values to floats
    matrix_list = [[float(x) for x in row] for row in matrix_list]

    # Convert the list to a NumPy array
    matrix = np.array(matrix_list)
    if dir == filename1:
        mutex1.release()
    elif dir == filename2:
        mutex2.release()
    elif dir == filename3:
        mutex3.release()
    return matrix

if __name__ == "__main__":
    PORT1 = 12340
    PORT2 = 12341
    PORT3 = 12342

    IP_ADDRESS1 = '127.0.0.1'
    IP_ADDRESS2 = '127.0.0.1'
    IP_ADDRESS3 = '127.0.0.1'
    #双机运行
    #建立连接
    server_establish(IP_ADDRESS1,PORT1,IP_ADDRESS2,PORT2,IP_ADDRESS3,PORT3)

    # Create threads for sending and receiving data
    receive_thread1 = threading.Thread(target=receive_data1)
    receive_thread2 = threading.Thread(target=receive_data2)
    receive_thread3 = threading.Thread(target=receive_data3)

    # command_joint([1,1,1,1],1)
    #指令发送线程
    send_thread1 = threading.Thread(target=send_command1)
    send_thread2 = threading.Thread(target=send_command2)
    send_thread3 = threading.Thread(target=send_command3)

    # Start threads
    receive_thread1.start()
    receive_thread2.start()
    receive_thread3.start()

    send_thread1.start()
    send_thread2.start()
    send_thread3.start()

    time.sleep(20)
    #结束命令#同时发送结束指令
    stop_threading()
    print('stop_threading')
    receive_thread1.join()
    receive_thread2.join()
    receive_thread3.join()

    send_thread1.join()
    send_thread2.join()
    send_thread3.join()

    #关闭server
    stopserver()
    sys.exit()