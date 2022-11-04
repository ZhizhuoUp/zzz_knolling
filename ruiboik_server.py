# echo-server.py
# see website: https://realpython.com/python-sockets/ for details in using socket
import socket
import math as m
import numpy as np
from turdf import *


# socket.socket() creates a socket object that supports the context manager type, 
# so you can use it in a with statement. There’s no need to call s.close()
# AF_INET is the Internet address family for IPv4. 
# SOCK_STREAM is the socket type for TCP, the protocol that will be used to transport messages in the network.
# Declare RealSense pipeline, encapsulating the actual device and sensors
#############################################################################
if __name__ == "__main__":
    HOST = "192.168.0.175"  # Standard loopback interface address (localhost)
    PORT = 8880  # Port to listen on (non-privileged ports are > 1023)
    real_pos_list = []
    Save_real_pos = False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        # It should be an integer from 1 to 65535, as 0 is reserved. Some systems may require superuser privileges if the port number is less than 1024.
        # associate the socket with a specific network interface
        s.listen()
        print(f"Waiting for connection...\n")
        conn, addr = s.accept()

        # The with statement is used with conn to automatically close the socket at the end of the block.
        with conn:
            print(f"Connected by {addr}")

            # or p.DIRECT for non-graphical version
            physicsClient = p.connect(1)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
            p.setGravity(0, 0, -9.8)
            planeId = p.loadURDF("plane.urdf")
            table_scale = 0.7
            table_surface_height = 0.625 * table_scale
            startPos = [0, 0, table_surface_height]
            startOrientation = p.getQuaternionFromEuler([0, 0, 0])
            boxId = p.loadURDF(filename + ".urdf", startPos, startOrientation, useFixedBase=1,
                               flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
            boxId2 = p.loadURDF("cube_small.urdf", [0.2, 0.2, table_surface_height], startOrientation, useFixedBase=0, flags=p.URDF_USE_SELF_COLLISION)
            boxId3 = p.loadURDF("table/table.urdf", [(0.5 - 0.16) * table_scale, 0, 0],
                                p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=1,
                                flags=p.URDF_USE_SELF_COLLISION, globalScaling=table_scale)
            p.changeDynamics(boxId, 7, lateralFriction=0.99, spinningFriction=0.02, rollingFriction=0.002)
            p.changeDynamics(boxId, 8, lateralFriction=0.99, spinningFriction=0.02, rollingFriction=0.002)
            p.changeDynamics(boxId2, -1, lateralFriction=0.99, spinningFriction=0.02, rollingFriction=0.002)
            reset(boxId)
            reset_real = np.asarray(real_cmd2tarpos([0.5, 0, 1, 0, 0.5]), dtype = np.float32)
            print(reset_real)
            conn.sendall(reset_real.tobytes())
            # time.sleep(10)
            for _ in range(200):
                p.stepSimulation()
                time.sleep(1 / 240)


            cmds_list = []

            ik_angles1 = p.calculateInverseKinematics(boxId, 9, targetPosition=[0.2, 0.2, table_surface_height+0.15],
                                                      maxNumIterations=200,
                                                      targetOrientation=p.getQuaternionFromEuler([0,1.57,0]))
            cmds_list.append(rad2cmd(ik_angles1[0:5]))
            ik_angles2 = p.calculateInverseKinematics(boxId, 9, targetPosition=[0.2, 0.2, table_surface_height + 0.01],
                                                      maxNumIterations=200,
                                                      targetOrientation=p.getQuaternionFromEuler([0, 1.57, 0]))
            cmds_list.append(rad2cmd(ik_angles2[0:5]))
            #
            # gripper_move
            #
            cmds_list.append([0.018, 0.018])
            #
            cmds_list.append(rad2cmd(ik_angles1[0:5]))

            ik_angles3 = p.calculateInverseKinematics(boxId, 9, targetPosition=[0.2, 0.205, table_surface_height+0.1],
                                                      maxNumIterations=200,
                                                      targetOrientation=p.getQuaternionFromEuler([0,1.57,0]))
            cmds_list.append(rad2cmd(ik_angles3[0:5]))

            ik_angles4 = p.calculateInverseKinematics(boxId, 9, targetPosition=[0.2, 0.205, table_surface_height + 0.02],
                                                      maxNumIterations=200,
                                                      targetOrientation=p.getQuaternionFromEuler([0, 1.57, 0]))

            cmds_list.append(rad2cmd(ik_angles4[0:5]))

            #
            # gripper_move
            #
            cmds_list.append([0, 0])
            #
            ik_angles4 = p.calculateInverseKinematics(boxId, 9, targetPosition=[0.2, 0.205, table_surface_height + 0.2],
                                                      maxNumIterations=200,
                                                      targetOrientation=p.getQuaternionFromEuler([0, 1.57, 0]))
            cmds_list.append(rad2cmd(ik_angles4[0:5]))

            p.setJointMotorControlArray(boxId, [7,8], p.POSITION_CONTROL, targetPositions=[0,0])



            ############################################


            for i in range(len(cmds_list)):
                if len(cmds_list[i]) == 5:
                    pos_sim = sim_cmd2tarpos(cmds_list[i])
                    pos_real = real_cmd2tarpos(cmds_list[i])
                    pos_real = np.asarray(pos_real, dtype=np.float32)
                    conn.sendall(pos_real.tobytes())
                    # p.setJointMotorControlArray(boxId, [0, 1, 2, 3, 4], p.POSITION_CONTROL, targetPositions=pos_sim)
                    p.setJointMotorControl2(boxId, 0, p.POSITION_CONTROL, targetPosition = pos_sim[0], maxVelocity= 3)
                    p.setJointMotorControl2(boxId, 1, p.POSITION_CONTROL, targetPosition=pos_sim[1], maxVelocity=3)
                    p.setJointMotorControl2(boxId, 2, p.POSITION_CONTROL, targetPosition=pos_sim[2], maxVelocity=3)
                    p.setJointMotorControl2(boxId, 3, p.POSITION_CONTROL, targetPosition=pos_sim[3], maxVelocity=3)
                    p.setJointMotorControl2(boxId, 4, p.POSITION_CONTROL, targetPosition=pos_sim[4], maxVelocity=3)
                elif len(cmds_list[i]) == 2:
                    pos_real = np.asarray(cmds_list[i], dtype=np.float32)
                    conn.sendall(pos_real.tobytes())
                    #gripper
                    p.setJointMotorControlArray(boxId, [7,8], p.POSITION_CONTROL, targetPositions=[cmds_list[i][0], cmds_list[i][1]])

                for _ in range(200):
                    p.stepSimulation()
                    time.sleep(1/240)

                real_pos = conn.recv(1024)
                real_pos = np.frombuffer(real_pos, dtype=np.float32)
                print(real_pos)
                if Save_real_pos == True:
                    real_pos_list.append(real_pos)





