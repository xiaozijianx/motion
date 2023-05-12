from djitellopy import tello
import time
from client import client_establish, send_data, receive_command, return_command
import threading

stop_receiving = False
take_photo = 0
IP_ADDRESS = '192.168.1.107'
PORT1 = 12340
PORT2 = 12341
PORT3 = 12342

#client_establish(IP_ADDRESS, PORT1)

# connect to tello
me = tello.Tello()
me.connect()
print(me.get_battery())
# tuen off and on streaming
#me.streamoff()
#me.streamon()
me.takeoff()
time.sleep(1)
me.move_up(50)
# get photo
#img = me.get_frame_read().frame


def recognize():
    global take_photo
    global img
    while True:
        #time.sleep(0.5)
        if take_photo == 1:
            pass 
        # 原本是发送take photo 拍照，处理然后讲结果存入json文件 这里可以替换为自己的逻辑
        # img, _ = videowithouttag2matrix(img, 1, 3, 3)

client_establish(IP_ADDRESS, PORT1)
send_thread = threading.Thread(target=send_data)
receive_thread = threading.Thread(target=receive_command)
photo_thread = threading.Thread(target=recognize)

send_thread.start()
receive_thread.start()
photo_thread.start()


while True:
    if stop_receiving:
        break
#    img = me.get_frame_read().frame
    velocity, take_photo, stop_receiving = return_command()
    #print(velocity)
    me.send_rc_control(-velocity[0], velocity[2],-velocity[1], 0)
#    img = me.get_frame_read().frame
    #me.send_rc_control(0, 0, -velocity[1], 0)
    #me.send_rc_control(0, velocity[2], 0, 0)

# connect to tello
# me = tello.Tello()
# me.connect()
# print(me.get_battery())
# # tuen off and on streaming
# me.streamoff()
# me.streamon()
# me.takeoff()
# time.sleep(3)
# me.move_up(100)

# take = 0

# def fly(xVal,yVal,zVal):
#     me.send_rc_control(xVal, -zVal, -yVal, 0)

# def take_picture():
#     me.get_frame_read().frame
#     pass

# if __name__ == "__main__":
#     while True:
#         me.fly()
#         if take == 1:
#             take_picture()
#         pass
