import utils.airsim.setup_path as setup_path
import airsim
import tempfile
import os
import json
import numpy as np
import cv2
import pprint
import time
from utils.airsim.Utils import getImagesResponse, getRGB_D, reponseTOrgb, responseTOpng, responseTOdepth
from scipy.spatial.transform import Rotation as R

environment_limits = {"Drone1": [8.5, -8.2, -3], "Drone2": [9, 13.2, -3],
                      "Drone3": [-17.4, 13.2, -3], "Drone4": [-16.4, -8.0, -3]}

orientations_deg = {"Drone1": [0, 0, 135], "Drone2": [0, 0, -135],
                    "Drone3": [0, 0, -45], "Drone4": [0, 0, 45]}


class MultiRotor:
    def __init__(self, name, init_position, vel):
        self.name = name
        self.X_init = init_position[0]
        self.Y_init = init_position[1]
        self.Z_init = init_position[2]
        self.vel = vel

        self.states = []

    def arm(self, client):
        # arm drone
        client.enableApiControl(True, self.name)
        client.armDisarm(True, self.name)
        # return client

    def take_off(self, client):
        # take off
        task = client.takeoffAsync(vehicle_name=self.name)
        return task

    def disarm(self, client):
        client.armDisarm(False, self.name)
        return client

    def moveTOpos(self, client, position, yaw=None):
        x_f = position[0] - self.X_init
        y_f = position[1] - self.Y_init
        z_f = position[2] - self.Z_init

        # print("received yaw", yaw)
        if yaw is not None:
            # print('moving drone')
            task = client.moveToPositionAsync(x_f, y_f, z_f, self.vel, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw), vehicle_name=self.name)
        else:
            task = client.moveToPositionAsync(x_f, y_f, z_f, self.vel, vehicle_name=self.name)
        return client, task

    def moveByVel(self, client, velocity, yaw=None, duration=1.0):
        x_velf = velocity[0]
        y_velf = velocity[1]

        if yaw is not None:
            print('moving drone')
            task = client.moveByVelocityAsync(x_velf, y_velf, 0.0, duration, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw), vehicle_name=self.name)
        else:
            task = client.moveByVelocityAsync(x_velf, y_velf, 0.0, duration, vehicle_name=self.name)

        return client, task

    def moveByVelZ(self, client, velocity, yaw=None, duration=1.0, zheight = 0):
        x_velf = velocity[0]
        y_velf = velocity[1]

        if yaw is not None:
            #print('moving drone')
            task = client.moveByVelocityZAsync(x_velf, y_velf, zheight, duration, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw), vehicle_name=self.name)
        else:
            task = client.moveByVelocityZAsync(x_velf, y_velf, zheight, duration, vehicle_name=self.name)
        return client, task

    def moveTOrelpos(self, client, position, yaw=None, offset = np.zeros(3)):
        x_f = position[0]
        y_f = position[1]
        z_f = position[2]

        if yaw is not None:
            print('moving drone')
            task = client.moveToPositionAsync(x_f, y_f, z_f, self.vel, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw), vehicle_name=self.name)
        else:
            task = client.moveToPositionAsync(x_f, y_f, z_f, self.vel, vehicle_name=self.name)
        return client, task

    def get_gt_3Dposition(self,client):
        state = self.get_state(client)
        global_position = self.get_position(state.position)
        return global_position


    def get_position(self, position):
        x_f = position.x_val + self.X_init
        y_f = position.y_val + self.Y_init
        z_f = position.z_val + self.Z_init

        return x_f, y_f, z_f

    def set_position(self, client, position, orientation):
        # set orientation
        pose = airsim.Pose(airsim.Vector3r(position[0], position[1], position[2]),
                           airsim.to_quaternion(orientation[0], orientation[1],orientation[2]))
        client.simSetVehiclePose(pose, True, vehicle_name=self.name)
        return client

    def get_rgb_d(self, client):
        responses = getImagesResponse(client, [self.name])
        frame, depth = getRGB_D(responses, [self.name])
        return frame, depth

    def get_state(self, client):
        # State reference from the drone started point
        # state = client.getMultirotorState(vehicle_name=self.name)
        state = client.simGetGroundTruthKinematics(vehicle_name=self.name)

        global_position = self.get_position(state.position)
        state.position.x_val = global_position[0]
        state.position.y_val = global_position[1]
        state.position.z_val = global_position[2]
        return state

    def get_focal_length(self, client):
        """
        :param client: airsim.VehicleClient
        :return: focal length of the camera [mm]
        """
        camera_name = "0"
        f = client.simGetFocalLength(camera_name, vehicle_name=self.name)
        return f

    def get_gps_data(self, client):
        gps_data = client.getGpsData(vehicle_name=self.name)
        return gps_data

    def get_init_position_and_orientation(self):
        next_pos = environment_limits[self.name]
        next_yaw = orientations_deg[self.name][2]
        return next_pos, next_yaw

    def get_extrinsic(self, client):
        # Get rotation matrix
        state = self.get_state(client)
        rotation = state.orientation
        quaternions = [rotation.x_val, rotation.y_val, rotation.z_val, rotation.w_val]
        rot = R.from_quat(quaternions)
        rot_matrix = rot.as_matrix()

        # Get translation
        translation = [[state.position.x_val],
                       [state.position.y_val],
                       [state.position.z_val]]
        E = np.append(rot_matrix, translation, axis=1)

        assert np.shape(E)[0] == 3 and np.shape(E)[1] == 4

        return translation, rot_matrix, E

    def update_state_info(self, client, i):
        # state_st = self.get_state(client)
        # position_st = self.get_position(state_st.kinematics_estimated.position)
        # orientation_st = state_st.kinematics_estimated.orientation
        # print('estimated state', state_st)

        state = client.simGetGroundTruthKinematics()
        # print('gt state', state)
        position = self.get_position(state.position)
        orientation = state.orientation
        # timestamp = state.timestamp

        pos_x = position[0]
        pos_y = position[1]
        pos_z = position[2]

        orient_w = orientation.w_val
        orient_x = orientation.x_val
        orient_y = orientation.y_val
        orient_z = orientation.z_val

        focal_length = self.get_focal_length(client)

        info_state = {'timestamp': i,
                      'focal_length': focal_length,
                      'pos_x': pos_x,
                      'pos_y': pos_y,
                      'pos_z': pos_z,
                      'orient_w': orient_w,
                      'orient_x': orient_x,
                      'orient_y': orient_y,
                      'orient_z': orient_z}
        self.states.append(info_state)

    def save_info_state(self, path):
        final_path = path + '/' + self.name + '/state_info.json'
        with open(final_path, 'w') as f:
            json.dump(self.states, f)
