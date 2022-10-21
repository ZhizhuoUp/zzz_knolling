from items import sort
import pybullet as p
import pybullet_data as pd
import os
import numpy as np
import random
import math
import cv2
import logging
from easy_logx.easy_logx import EasyLog
from urdf_parser_py.urdf import URDF
from easy_logx.easy_logx import EasyLog
import os
import numpy as np
from stl import mesh

logger = EasyLog(log_level=logging.INFO)

class Arm_env():
    
    def __init__(self, max_step, is_render=True, num_objects=1, x_grasp_accuracy=0.2, y_grasp_accuracy=0.2,
                 z_grasp_accuracy=0.2, order_flag = 'center'):

        self.kImageSize = {'width': 480, 'height': 480}

        self.step_counter = 0

        self.urdf_path = 'urdf'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render

        self.ik_low = [-1.57, -1.57, -1.57, -1.57, -1.57, -10, -10]
        self.ik_high = [1.57, 1.57, 1.57, 1.57, 1.57, 10, 10]

        self.low_scale = np.array([0.05, -0.15, 0.005, - np.pi / 2, 0])
        self.high_scale = np.array([0.3, 0.15, 0.05, np.pi / 2, 0.4])
        self.low_act = -np.ones(5)
        self.high_act = np.ones(5)
        self.x_low_obs = self.low_scale[0]
        self.x_high_obs = self.high_scale[0]
        self.y_low_obs = self.low_scale[1]
        self.y_high_obs = self.high_scale[1]
        self.z_low_obs = self.low_scale[2]
        self.z_high_obs = self.high_scale[2]

        self.x_grasp_interval = (self.x_high_obs - self.x_low_obs) * x_grasp_accuracy
        self.y_grasp_interval = (self.y_high_obs - self.y_low_obs) * y_grasp_accuracy
        self.z_grasp_interval = (self.z_high_obs - self.z_low_obs) * z_grasp_accuracy

        self.obs = np.zeros(19)
        self.table_boundary = 0.05
        self.max_step = max_step

        self.friction = 0.99
        self.num_objects = num_objects
        self.order_flag = order_flag
        # self.action_space = np.asarray([np.pi/3, np.pi / 6, np.pi / 4, np.pi / 2, np.pi])
        # self.shift = np.asarray([-np.pi/6, -np.pi/12, 0, 0, 0])
        self.ik_space = np.asarray([0.3, 0.4, 0.06, np.pi])  # x, y, z, yaw
        self.ik_space_shift = np.asarray([0, -0.2, 0, -np.pi / 2])

        self.slep_t = 1 / 50
        self.joints_index = [0, 1, 2, 3, 4, 7, 8]
        # 5 6 9不用管，固定的！
        self.init_joint_positions = [0, 0, -1.57, 0, 0, 0, 0, 0, 0, 0]

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.camera_parameters = {
            'width': 640.,
            'height': 480,
            'fov': 42,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.25, 0, 0.05],
            distance=0.4,
            yaw=90,
            pitch=-50.5,
            roll=0,
            upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] /
                   self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        p.resetDebugVisualizerCamera(cameraDistance=0.7,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0.4])
        p.setAdditionalSearchPath(pd.getDataPath())
        self.reset()

    def reset(self):

        self.step_counter = 0

        p.resetSimulation()
        p.setGravity(0, 0, -10)

        # Draw workspace lines
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])

        baseid = p.loadURDF(os.path.join(self.urdf_path, "base.urdf"), basePosition=[0, 0, -0.05], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1.urdf"),
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(os.path.join(self.urdf_path, "table/table.png"))
        p.changeDynamics(baseid, -1, lateralFriction=self.friction)
        p.changeDynamics(self.arm_id, 7, lateralFriction=self.friction)
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.friction)
        p.changeVisualShape(baseid, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=textureId)

        self.obj_idx = []
        for i in range(self.num_objects):
            rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs), random.uniform(self.y_low_obs, self.y_high_obs),
                       0.01]
            rdm_ori = [0, 0, random.uniform(-math.pi / 2, math.pi / 2)]
            self.obj_idx.append(p.loadURDF(os.path.join(self.urdf_path, "lego_cube/urdf/cube%d.urdf" % i), basePosition=rdm_pos,
                                           baseOrientation=p.getQuaternionFromEuler(rdm_ori),
                                           flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
            p.changeDynamics(self.obj_idx[i], -1, lateralFriction=self.friction, spinningFriction=0.02,
                             rollingFriction=0.002)
            logger.debug(f'this is the urdf id: {self.obj_idx}')

        # ! initiate the position
        # TBD use xyz pos to initialize robot arm (IK)
        p.setJointMotorControlArray(self.arm_id, [0, 1, 2, 3, 4, 7, 8], p.POSITION_CONTROL,
                                    targetPositions=[0, -0.48627556248779596, 1.1546790099090924, 0.7016159753143177, 0,
                                                     0, 0],
                                    forces=[10] * 7)

        xyz_list, items_pos_list, items_ori_list = self.get_data()
        print(xyz_list)
        items_sort = sort()
        cube_2x2, cube_2x3,cube_2x4, pencil, others = items_sort.judge(xyz_list)

        # self.reorder(xyz_list, items_pos_list, items_ori_list)

        # while 1:
        #     p.stepSimulation()

    def get_data(self):

        #! Generate the pos, dimension and orin of objects randomly.
        
        items_pos_list, items_ori_list = [], []
        for i in range(len(self.obj_idx)):
            cube_pos = np.asarray(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
            cube_ori = np.asarray(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[0])[1]))
            items_pos_list.append(cube_pos)
            items_ori_list.append(cube_ori)
        items_pos_list = np.asarray(items_pos_list, dtype=np.float32)
        items_ori_list = np.asarray(items_ori_list, dtype=np.float32)

        names = globals()
        xyz_list = []
        for i in range(self.num_objects):
            names[f'cube_{i}_dimension'] = mesh.Mesh.from_file('urdf/lego_cube/meshes/cube%d.STL' % i)
            xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        xyz_list = np.asarray(xyz_list, dtype=np.float32)

        return xyz_list, items_pos_list, items_ori_list

    def reorder(self, xyz_list, items_pos_list, items_ori_list):

        gap = 0.01
        # print(names['self.cube_1_dimension'])
        
        if self.order_flag == 'center':
            
            #! calculate the number of row and column
            if self.num_objects % 2 != 0:
                num_cube = self.num_objects + 1
            fac = [] # 定义一个列表存放因子
            for i in range(1, num_cube):
                if num_cube % i == 0:
                    fac.append(i)
                    continue
                else:
                    pass
            # print(fac)
            num_row = int(random.choice(fac))
            print(num_row)
            num_column = int(num_cube / num_row)
            print(num_column)

            config_array = np.arange(num_cube).reshape(num_row, num_column)
            print(config_array)

            print(items_pos_list)
            print('aaa')

            i = 0
            for m in range(num_row):
                for n in range(num_column):

                    print(config_array[m, n])
                    index = config_array[m, n]
                    if index == self.num_objects:
                        print('over')
                        break
                    
                    print(items_pos_list[index])


                    i += 1

            #! draw the default boundary
            x_range = np.sum(xyz_list[:, 0]) + gap * (self.num_objects - 1)
            y_range = np.sum(xyz_list[:, 1]) + gap * (self.num_objects - 1)
            x_low = (self.x_high_obs + self.x_low_obs) / 2 - x_range / 2
            x_high = (self.x_high_obs + self.x_low_obs) / 2 + x_range / 2
            y_low = (self.y_high_obs + self.y_low_obs) / 2 - y_range / 2
            y_high = (self.y_high_obs + self.y_low_obs) / 2 + y_range / 2

            p.addUserDebugLine(
                lineFromXYZ=[x_low, y_low, self.z_low_obs],
                lineToXYZ=[x_high, y_low, self.z_low_obs])
            p.addUserDebugLine(
                lineFromXYZ=[x_low, y_low, self.z_low_obs],
                lineToXYZ=[x_low, y_high, self.z_low_obs])
            p.addUserDebugLine(
                lineFromXYZ=[x_high, y_low, self.z_low_obs],
                lineToXYZ=[x_high, y_high, self.z_low_obs])
            p.addUserDebugLine(
                lineFromXYZ=[x_low, y_high, self.z_low_obs],
                lineToXYZ=[x_high, y_high, self.z_low_obs])

            items_ori_list[:, 2] = 0
            logger.info('reset the yaw of all cubes')
            # print(items_ori_list)



        # return new_xyz, new_cube_pos, new_cube_ori

if __name__ == '__main__':
    
    env = Arm_env(max_step=3, is_render=True, num_objects=5, order_flag = 'center')
    # env.reorder()