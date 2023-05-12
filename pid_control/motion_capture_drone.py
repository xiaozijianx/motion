# april tag detection
# import apriltag
# import pupil_apriltags as apriltag     # for windows
import socket
import cv2
import numpy as np
# import sys
import cvzone
# from djitellopy import tello
import time
# from color_recognize import video2matrix
from djitellopy import tello
import sys
sys.path.insert(0,'D:\distributed-sensing')
from transform import transform
# from color_recognize import videowithouttag2matrix
# cap = cv2.VideoCapture(0)

# hi, wi = 480, 640
# PID control
#                    P  I  D
#             P:ratio to drone speed
#             D: derivative term to redce the speed cause by momentum
# xPID = cvzone.PID([0.11, 0, 0.3], wi//2)
# yPID = cvzone.PID([0.27, 0, 0.2], hi//2, axis=1)
# # when it comes closer when it comes backward
# zPID = cvzone.PID([0.015, 0, 0.9], 2500, limit=[-15, 15])
# # zPID = cvzone.PID([0.0015, 0, 0.007], 25000, limit=[-15, 15])


myPlotX = cvzone.LivePlot(yLimit=[-100, 100], char='X')
myPlotY = cvzone.LivePlot(yLimit=[-100, 100], char='Y')
myPlotZ = cvzone.LivePlot(yLimit=[-100, 100], char='Z')

time.sleep(2.0)


# me = tello.Tello()
# me.connect()
# print(me.get_battery())
# # # tuen off and on streaming
# me.streamoff()
# me.streamon()
# me.takeoff()
# time.sleep(1)
# me.move_up(150)





def pid_control(tello_loc, target_loc):
    tello_x = tello_loc[0]
    tello_y = tello_loc[1]
    tello_z = tello_loc[2]
    target_x = target_loc[0]
    target_y = target_loc[1]
    target_z = target_loc[2]
    # 傳送目標位置到PID函式進行初始化
    K_LIMIT = 20
    xPID = cvzone.PID([0.11, 0, 0.01], target_x, limit=[-K_LIMIT, +K_LIMIT])
    yPID = cvzone.PID([0.3, 0, 0.01], target_y, axis=1, limit=[-K_LIMIT, +K_LIMIT])
    # when it comes closer when it comes backward
    zPID = cvzone.PID([0.15, 0, 0.01], target_z, limit=[-K_LIMIT, +K_LIMIT])


    # 利用PID計算velocity
    xVal = int(xPID.update(tello_x))  # error distance btween center
    yVal = int(yPID.update(tello_y))  # error distance btween center
    zVal = int(zPID.update(tello_z))  # error distance btween center
    # rc_control(x,-z,-y)
    return xVal, yVal, zVal


# tello_x = 250
# tello_y = 250
# tello_z = 250
# target_x = 90
# target_y = 80
# target_z = 70


if __name__ == "__main__":
    me = tello.Tello()
    me.connect()
    print(me.get_battery())
    # # tuen off and on streaming
    me.streamoff()
    me.streamon()
    me.takeoff()
    time.sleep(1)
    me.move_up(150)
    HOST, PORT = "127.0.0.1", 8888
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    while True:
        s.send(b'Hello')
        print('hello')
        # print("wait for server")
        id_locations_str = s.recv(1000).decode()
        # print(id_locations_str)
        N_rigids = 1
        locations = np.zeros((N_rigids,3))
        for line in id_locations_str.splitlines():
            words = line.split()
            rigid_name = words[0]
            id = int(rigid_name[5]) # 'rigidX'
            if id == 3:
                locations[0] = np.genfromtxt(words[1:], dtype=float)
        print(123)
        img = me.get_frame_read().frame
        # print(123)
        #_, img = cap.read()
        # img = cv2.resize(img, (640, 480))

        # 控制
        xVal = 0
        yVal = 0
        zVal = 0
        # 傳入tello位置及target的位置
        # if tello_x != target_x:
        #     tello_x -= 10
        # if tello_y != target_y:
        #     tello_y -= 10
        # if tello_z != target_z:
        #     tello_z -= 10
        
        # 设置target position
        pos_grid = (8,8)
        pos_local = transform.grid2local(pos_grid)
        pos_world = transform.local2world(pos_local)
        # target_loc = np.array([pos_world[0], pos_world[1], -1600])
        # target_loc = np.array([pos_world[0], pos_world[1], -800])
        target_loc = np.array([pos_world[0], pos_world[1], 0])
        # target_loc = np.array([pos_world[0], pos_world[1], 1000]) # too far away
        
        xVal, yVal, zVal = pid_control(locations[0], target_loc)


        imgPlotX = myPlotX.update(xVal)
        imgPlotY = myPlotY.update(yVal)
        imgPlotZ = myPlotZ.update(zVal)

        # imageStacked = cvzone.stackImages(
            # [img, imgPlotX, imgPlotY, imgPlotZ], 2, 0.75)
        #cv2.imshow("Stacked", imageStacked)

        me.send_rc_control(-xVal, zVal, -yVal, 0)
        # me.send_rc_control(0, 0, -yVal, 0)
        # me.send_rc_control(0, zVal, 0, 0)
        #me.send_rc_control(-xVal, 0, 0, 0)
        # cv2.imshow("MC_test", imageStacked)
        # img_recognize = videowithouttag2matrix(img, 1,3, 3)
        cv2.imshow("MC_test", img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            me.end()
            me.streamoff()
            break
        if cv2.waitKey(5) & 0xFF== ord('w'):
            cv2.imwrite('test1.jpg',img)
            print('screenshot')

    cv2.destroyAllWindows()
