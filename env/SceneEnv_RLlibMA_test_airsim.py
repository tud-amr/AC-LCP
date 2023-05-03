"""
Scene environment for information gathering using RL
---
This file intends to put all the configuration options together
"""


import random
import time
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from typing import NamedTuple
from utils import auxFunctions
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import Polygon
import scipy.stats as distrib
import math
from gym import spaces
from ray.rllib.utils.spaces.repeated import Repeated
import matplotlib.gridspec as gridspec
from celluloid import Camera


import csv
from collections import OrderedDict
from env.SceneEnv_RLlibMA import SceneEnv as MAScene_Base
from utils.approx_perception import Pedestrian

# import importlib
# importlib.reload(Scene)

import os
import gc


## SIMULATED DRONE
from utils.airsim.pedestrians import get_name_pedestrians, update_gt2d_pedestrian
from utils.airsim.settings_airsim import Configuration
import airsim
from utils.airsim.drone import MultiRotor
from utils.airsim.image_utils import getResponseImages, save_images, getResponseImagesIdx
from utils.airsim.Visualization import visualize, visualize_4Plots


# Constants (TODO NOT NOW to edit them from the main file)
MAXTARGETS = 8
MAX_STEPS = 100
SIDE = 25 #25.0 #25 #8.0 #TODO
SIDEx = 50
SIDEy = 50
OBS_BOUNDS = 16  # TODO without
DIM_TARGET = 0.6
DIM_ROBOT = DIM_TARGET
MINDIST = 2.3 * DIM_TARGET
BELIEF_THRESHOLD = 0.95 #1.0 #0.95


def trainEvlogs2csv(folder_file_name,csvDict, n_episode):
    fieldnames = list(csvDict.keys())
    if n_episode == 1 and False:
        #print('this happens')
        with open(folder_file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(csvDict)
    else:
        with open(folder_file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(csvDict)

class TargetPlot(NamedTuple):
    shirt_color: np.array
    shirt: mpl.patches.Wedge
    face: mpl.patches.Wedge


class RobotPlot(NamedTuple):
    color: np.array
    shape: mpl.patches.Rectangle
    fov: mpl.patches.Wedge

#class SceneEnv(gym.Env, rllib.MultiAgentEnv):
class SceneEnv(MAScene_Base, ):
    def __init__(self, config):
        MAScene_Base.__init__(self, config)

        self.robot_target_assignment = True and self.heuristic_target_order

        ###IMPORTANT####
        self.multiagent_policy = False

        # GYM and RL variables
        # Action is the displacement of the robot
        # SCENE1D in this case it is only action in X
        self.action_space_mapping = np.array([3, 3, 1]) # Scaling action factor ## IF not teleporting use 3 instead of 2

        self.deglossed_obs = False

        self.action_space=[]
        self.observation_space=[]
        for i in range(self.nrobots):
            # individual robot action space
            self.robot_action_space = spaces.Box(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]), shape=(3,),
                                            dtype=np.float32)
            # Append individual robot action space
            self.action_space.append(self.robot_action_space)

            # Individual robot observation space
            self.target_space = spaces.Dict({
                "location": spaces.Box(np.array([-2*np.sqrt(SIDEx*SIDEx+SIDEy*SIDEy), -2*np.sqrt(SIDEx*SIDEx+SIDEy*SIDEy), -np.pi]), np.array([2 * np.sqrt(SIDEx*SIDEx+SIDEy*SIDEy), 2 * np.sqrt(SIDEx*SIDEx+SIDEy*SIDEy), np.pi]), shape=(3,), dtype=np.float32),
                "belief": spaces.Box(np.array([0.0]), np.array([1.0+1e-10]), shape=(1,), dtype=np.float32),
                "measurement": spaces.Box(np.array([0.0]), np.array([1.0+1e-10]), shape=(1,), dtype=np.float32),
                "velocity": spaces.Box(np.array([-np.inf,-np.inf]), np.array([np.inf,np.inf]), shape=(2,), dtype=np.float32),
                "tracked": spaces.Box(np.array([0.0]), np.array([1.0]), shape=(1,), dtype=np.float32)
            })
            self.robot_space = spaces.Dict({
                "location": spaces.Box(np.array([-np.sqrt(SIDEx * SIDEx+SIDEy * SIDEy), -np.sqrt(SIDEx * SIDEx+SIDEy * SIDEy), -np.pi]),
                                       np.array([np.sqrt(SIDEx * SIDEx+SIDEy * SIDEy), np.sqrt(SIDEx * SIDEx + SIDEy * SIDEy), np.pi]),
                                       shape=(3,), dtype=np.float32),
               #"velocity": spaces.Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]), shape=(2,),dtype=np.float32),
            })
            # Observation is a list of targets.
            self.multiple_target_space = Repeated(self.target_space, max_len=self.MAX_TARGETS)

            self.total_target_space = spaces.Dict({"unidentified": self.multiple_target_space, "identified": self.multiple_target_space,
                                                      "all": self.multiple_target_space})
            # add observations of other robots
            self.total_robot_space = Repeated(self.robot_space, max_len=self.MAX_ROBOTS-1)

            # target observations deglossed into identified / unidentified and all.
            if self.deglossed_obs:
                if self.multiagent_policy:
                    self.robot_observation = spaces.Dict({"targets": self.total_target_space, "robots": self.total_robot_space})
                else:
                    self.robot_observation = spaces.Dict({"targets": self.total_target_space})
                self.observation_space.append(self.robot_observation)
            else:
                if self.multiagent_policy:
                    self.robot_observation = spaces.Tuple((self.multiple_target_space, self.total_robot_space))
                else:
                    self.robot_observation = self.multiple_target_space
                self.observation_space.append(self.robot_observation)

        self.action_space = spaces.Tuple(self.action_space)
        self.observation_space = spaces.Tuple(self.observation_space)

        # Variables for plotting
        self.is_plotting_scene = False
        self.is_plotting_beliefs = False
        self.is_plotting_images = False
        self.render_enabled = False

        self.delta_t = 0.25
        self.high_level_delta_t = 0.25
        self.lh_ratio = int(self.high_level_delta_t/self.delta_t)
        self.last_action = None

        self.robot_target_collision = False
        self.robot_robot_occlusion = True

        if self.env_mode == 'cte_vel':
            self.target_maxvel = 1  # max target velocity 1m/s

        if self.env_mode == 'sf_goal':
            self.socialForce_params = {
                'desvel_mean': 1.34,
                'desvel_std': 0.26,
                'potentials': {'target_cte': 21.1, #2.1 #NUMBERS FOR CORL
                               'target_sigma': 2.0, #0.3 #NUMBERS FOR CORL
                               'wall_cte': 10,
                               'wall_R': 0.6},  #0.2
                'delta_t': 2,
                'tao': 0.5,
                'psi': 100,  # pedestrian angle of sight
                'c': 0.5,  # influence outside angle of sight
            }
        if 'performance_test_logs' in config:
            self.log_folder = config['performance_test_logs']
            self.logs=True
        else:
            self.log_folder = []
            self.logs = False

        if self.test:
            if "load_scn" in config and config["load_scn"]:
                self.load_scn = config["load_scn"]
                self.load_scn_folder = config["load_scn_folder"]
                self.scenario_loaded_batch = self.load_scn_batch()
            else:
                self.load_scn = False
                self.scenario_loaded_batch = []
        else:
            self.load_scn = False
            self.scenario_loaded_batch = []


        # domain randomization (ntargets not included)
        self.random_beliefs = config['random_beliefs']
        self.random_static_dynamic = config['random_static_dynamic']
        if self.random_static_dynamic:
            self.static_ratio = 0.25

        self.visibility_hist = False or (self.test and True)

        self.save_video = False or (self.test and True)

        # robots are agents in RLlib
        self.num_agents = self.nrobots
        self.agent_ids = list(range(self.num_agents))
        self.observation_space_dict = self._make_dict(self.observation_space)
        self.action_space_dict = self._make_dict(self.action_space)

        self.observation_space = self.observation_space[0]
        self.action_space = self.action_space[0]

        # enable/disable realistic drone dynamics
        self.realistic_Dengine = "airsim" # none / dummy_env / airsim
        if self.realistic_Dengine=="dummy_env":
            from realistic_drone.py_g2g_drone_sim.test_gym.BasicDroneDynamics import DroneEnv as droneEngine
            self.drone_engines = [droneEngine(ndrones = 1, ndynamic_obstacles =0) for i in range(self.nrobots)]

        self.dyn_sigma=0.0

        #self.init_scene(nrobots=self.nrobots, ntargets=self.ntargets)

        #self.time_aux = time.time()
        self.simulated_perception = "airsim" # dummy / simulated /airsim /simulated_airsim TODO
        if self.simulated_perception == "simulated" or self.simulated_perception=="simulated_airsim":
            #self.yolo_model = create_trained_yolo()
            self.unrealPedestrianData = [Pedestrian(1,real_class+1,self.number_of_classes) for real_class in range(self.number_of_classes)]
        if self.simulated_perception == "simulated":
                for pedestrian in self.unrealPedestrianData:
                    pedestrian.load_probas()
        if self.simulated_perception == "airsim":
            self.unrealPedestrianData = [Pedestrian(1, real_class + 1, self.number_of_classes, load_imgs=False, load_poses=False) for real_class in range(self.number_of_classes)]
            self.ped_class_dict = {'A': 0, 'B': 1}
        if self.simulated_perception == "simulated_airsim" or self.simulated_perception == "airsim":
                import env.settings as settings
                self.yolo = settings.yolo
                self.init_airsim_config()


    def init_airsim_config(self):
        # Defining settings
        velocity = 0.5 #3#0.6 # 3 ## 0.5 if the environment goes 6 times slower
        save_mode = 'start'  # 'start', 'wait'
        #frames_to_capture = 100 ## NOT USEFUL
        name_experiment = 'airsim_images'
        self.pedestrian_model = 'Louise' #James / Lixin / Louise / Damian
        img_types = 'RGB'  # 'RGB' , 'RGB-D', 'RGB-DS'
        drone1 = MultiRotor("Drone1", [0, 0, 0], velocity)
        drone2 = MultiRotor("Drone2", [0, 0, 0], velocity)
        drone3 = MultiRotor("Drone3", [0, 0, 0], velocity)
        drone4 = MultiRotor("Drone4", [0, 0, 0], velocity)
        self.dronelist = [drone1]#, drone2]#, drone3, drone4]
        self.drone_names = [drone.name for drone in self.dronelist]

        self.config = Configuration(img_types, None, save_mode, visualize_images=True, vis_pedestrian_2dGT=True,
                               name_experiment=name_experiment, external=False, uavs=self.dronelist)
        # Initializing Vehicle of AirSim in Unreal
        self.clients = [airsim.MultirotorClient() for i in range(len(self.dronelist))]

        self.cameras_names = self.config.camera_names

        print('-> External cameras defined {} in MODE: {} '.format(self.cameras_names, self.config.mode))
        # -------------Connect AirSim with Unreal--------------
        self.client_ref = self.clients[0]

        for client in self.clients:
            client.confirmConnection()

        struct_ref = 'BP_P_'  # reference structure in Unreal to look for the pedestrians
        self.name_pedestrians = get_name_pedestrians(self.clients[0], struct_ref, segmentation=self.config.semantic_segmentation)

        print('-> Pedestrians found in the scene: {}'.format(self.name_pedestrians))
        print("Number of pedestrians:", len(self.name_pedestrians))

        ped_pose = self.clients[0].simGetObjectPose(self.name_pedestrians[0])
        self.z_ref_ped = ped_pose.position.z_val
        self.z_ref_drone = [-2.4,-2.4]#-3.4,-4.4,-5.4]
        self.z_drone_curr = [-2.4,-2.4]#-3.4,-4.4,-5.4]

        self.offset_envCenter = np.array([0, 0])#np.array([16.4-12.45, 10.6-8.0])# np.array([16.4-12.45, 10.6-8.0])

        self.drone_ref = self.dronelist[0]

        for i in range(len(self.dronelist)):
            self.clients[i].simAddDetectionFilterMeshName(camera_name="0", image_type=airsim.ImageType.Scene, mesh_name="BP_P*",
                                             vehicle_name=self.dronelist[i].name)

        # initialise clock for control freq
        self.clock_last_action = time.time()

        ## arm / take off drones
        task = None
        for i in range(len(self.dronelist)):
            self.dronelist[i].arm(self.clients[i])
            # take off
            task = self.dronelist[i].take_off(self.clients[i])
            task.join()

    # FUNCTION FOR GYM AND RL
    def reset(self, replace_targets=True):
        """
        Resets and samples a new initial configuration of the environment for the new episode if necessary.
        If we want to save a video, then last episode's video is generated.
        """
        # Clear lists and information from the last episode
        self.targets.clear()
        self.targets_ID.clear()
        self.robots.clear()
        self.robots_fov = []
        self.beliefs.clear()
        self.episode_beliefs.clear()
        self.episode_visibility.clear()
        self.episode_poses.clear()
        self.episode_entropy = []
        self.episode_reward = 0

        # Set a new number of targets and robots in the new episode
        if len(self.nmaxtargets) == 1:
            self.ntargets = self.nmaxtargets[-1]
        else:
            self.ntargets = np.random.randint(self.nmaxtargets[0], self.nmaxtargets[
                1] + 1) if self.test == False else self.nmaxtargets  # np.random.randint(1,self.MAX_TARGETS+1)

        if len(self.nmaxrobots) == 1:
            self.nrobots = self.nmaxrobots[-1]
        else:
            self.nrobots = np.random.randint(self.nmaxrobots[0], self.nmaxrobots[
                1] + 1) if self.test == False else self.nmaxrobots  # np.random.randint(1,self.MAX_TARGETS+1)

        # Initialize episode number and success log
        self.nepisode += 1
        self.success = 0

        if self.heuristic_target_order:
            self.targets_id_order = [[] for r in range(self.nrobots)]

        # If previously specified, use saved images to generate a .gif of the episode
        if self.save_video and self.nepisode > 3:
            camfolder = self.video_foldername + 'camera/'
            belfolder = self.video_foldername + 'beliefs/'
            statefolder = self.video_foldername + 'state/'
            # videosource = ['camera', 'beliefs', 'state']
            videosource = ['beliefs', 'state']

            def make_gif(frame_folder, outfolder, videosource):
                import glob
                from PIL import Image
                imagelist = glob.glob(f"{frame_folder}/*.png")
                sortedimagelist = sorted(imagelist, key=lambda x: int(x.split('/')[-1][:-4]))
                frames = [Image.open(image) for image in sortedimagelist]
                frame_one = frames[0]
                frame_one.save(outfolder + videosource + ".gif", format="GIF", append_images=frames,
                               save_all=True, duration=250, loop=0)  # duration=250, loop=0)

            # for i, video_folder in enumerate([camfolder, belfolder, statefolder]):
            for i, video_folder in enumerate([belfolder, statefolder]):
                make_gif(video_folder, self.video_foldername, videosource[i])

        # Reset the state of the environment to an initial state with the specified nrobots and ntargets
        # TODO: DONE update this function to randomize the number of robots as well.
        self.init_scene(nrobots=self.nrobots, ntargets=self.ntargets, replace_targets = replace_targets)

        # Track target visibility (mostly used to decide whether a new observation is obtained from a target or not)
        self.visibility = self.check_visibility()
        self.detected_pedestrian_dict_list = [detectedPedestriansImg(self.clients[r], self.config, r) for r in range(self.nrobots)]
        # FROM HERE!!
        # Generate new probability estimates over each target over the environment. TODO: DONE update to multirobot
        for r in range(self.nrobots):
            for t in range(self.ntargets):
                self._observation[t, r] = self.observation(t, r)

        # Generate a new observation over the environment. TODO: DONE update to multirobot
        obs = self._obs()

        # return obs
        obs_dict = self._make_dict(obs)

        gc.collect()  ### make sure it works
        return obs_dict

    def init_scene(self, nrobots=1, ntargets=1, replace_targets=True):

        # Parameter initialization
        self.steps = 0
        self.hl_steps = 0
        self.last_action = None
        self.ntargets = ntargets
        if self.env_mode == "airsim":
            ntargets = len(self.name_pedestrians)
            self.ntargets = len(self.name_pedestrians)
            self.airsim_timestamp_t1 = []
            self.airism_timestamp = []

        self.colors = auxFunctions.get_cmap(self.number_of_classes)
        self.targets = [np.zeros(3) for i in range(ntargets)]
        self.targets_t1 = [np.zeros(3) for i in range(ntargets)]
        self.targets_vel = [np.zeros(2) for i in range(ntargets)]
        self.targets_desvel = [0 for i in range(ntargets)]
        self.targets_maxvel = [0 for i in range(ntargets)]
        self.targets_goal = [np.zeros(2) for i in range(ntargets)]
        self.target_colors = np.random.rand(3, self.ntargets)
        self.nrobots = nrobots
        if self.env_mode == "airsim":
            self.nrobots = len(self.dronelist)
            nrobots = len(self.dronelist)
            self.num_agents = self.nrobots
            self.agent_ids = list(range(self.num_agents))
            if self.heuristic_target_order:
                self.targets_id_order = [[] for r in range(self.nrobots)]
        self.robots = [np.zeros(3) for r in range(self.nrobots)]
        self.robots_fov = 45 * np.ones(nrobots)
        self._observation = np.zeros((self.ntargets,self.nrobots,self.number_of_classes))
        if self.visibility_hist:
            self.obs_histogram = []
            self.obs_tracked_histogram = []
            self.tracked_histogram = []

        # Initial beliefs  # No need to be saved
        # Centralized beliefs (only one vector of beliefs)
        if self.random_beliefs:
            rand_distribution = np.random.rand(ntargets, self.number_of_classes)
            rand_distribution = rand_distribution / np.sum(rand_distribution,1)[:,None]
            self.beliefs = [rand_distribution for r in range(1)]
            self.num_targets_tracked = [np.any(self.beliefs[r]>0.95,1) for r in range(1)]
            self.targets_correct_classification = [np.zeros(ntargets, dtype=int) for r in range(nrobots)]
            for r in range(self.nrobots):
                for t in range(self.ntargets):
                    if self.num_targets_tracked[0][t]:
                        if self.beliefs[0][t][self.targets_ID[t]]>0.95:
                            self.targets_correct_classification[0][t] = 1
                        else:
                            self.targets_correct_classification[0][t] = -1
        else:
            values = 1.0 / self.number_of_classes
            self.beliefs = [values * np.ones((ntargets, self.number_of_classes)) for r in range(nrobots)]
            self.num_targets_tracked = [np.zeros(ntargets, dtype=bool) for r in range(nrobots)]
            self.targets_correct_classification = [np.zeros(ntargets, dtype=int) for r in range(nrobots)]

        if self.robot_target_assignment:
            self.target_assigned = [False for t in range(ntargets)]

        assert self.dimensions == 3  #otherwise bad

        if self.dimensions == 3 and (not self.load_scn or self.nepisode-2 > len(self.scenario_loaded_batch)):
            positions, _, _, _ = auxFunctions.randomPositionsProximity(N=ntargets + nrobots, side=SIDEy-2, dmin=MINDIST, # changed from SIDE TO SIDE-2
                                                                       dmax= np.sqrt(SIDEx*SIDEx+SIDEy*SIDEy))
            if self.env_mode == 'sf_goal':
                goals, _, _, _ = auxFunctions.randomPositionsProximity(N=ntargets, side=SIDEy,
                                                                       dmin=MINDIST,
                                                                       dmax=np.sqrt(SIDEx*SIDEx+SIDEy*SIDEy))
            random.shuffle(positions)
            for t in range(ntargets):
                self.targets[t][0:2] = np.copy(positions[t])
                self.targets[t][2] = np.random.uniform(0, 360)
                self.targets_ID.append(random.randint(0, self.number_of_classes - 1))

                if self.env_mode == 'cte_vel':
                    max_coordinate_vel = np.clip(np.random.normal(self.target_maxvel,0.2),-1.5,1.5)#(np.random.rand()*2 - 1) * self.target_maxvel
                    vel = np.random.rand(2) * max_coordinate_vel * np.random.choice([-1,1],2)
                    if self.random_static_dynamic: vel = vel * np.random.choice([0,1],p = [self.static_ratio, 1-self.static_ratio])
                    angle = np.angle(vel[0] + vel[1]*1j, deg=True)
                    if angle < 0:
                        angle += 360
                    self.targets_vel[t] = vel
                    self.targets[t][2] = angle

                if self.env_mode == 'sf_goal':
                    #max_coordinate_vel = (np.random.rand()*2 - 1) * self.target_maxvel
                    self.targets_desvel[t] = np.random.normal(loc = self.socialForce_params['desvel_mean'], scale = self.socialForce_params['desvel_std'])
                    self.targets_maxvel[t] = self.targets_desvel[t]*1.3
                    self.targets_goal[t][0:2] = np.copy(goals[t])
                    direction = (self.targets_goal[t] - self.targets[t][0:2]) / np.linalg.norm(self.targets_goal[t] - self.targets[t][0:2])
                    angle = np.angle(direction[0] + direction[1]*1j, deg=True)
                    if angle < 0:
                        angle += 360
                    self.targets_vel[t] = np.zeros(2) #direction*self.targets_desvel[t]
                    self.targets[t][2] = angle

                if self.env_mode == 'airsim':
                    if replace_targets:
                        # set initialised target poses in airsim
                        set_pose(self.client_ref, self.name_pedestrians[t],[self.targets[t][0],self.targets[t][1],self.z_ref_ped],[0,0,self.targets[t][2]],self.offset_envCenter)
                    self.airsim_timestamp_t1.append(time.time())
                    # associate core targetIDs to airsim textures
                    # ped_class = self.ped_class_dict[self.name_pedestrians[t].split('_')[2].split(self.pedestrian_model)[1][0]]
                    ped_class = self.ped_class_dict[
                        self.name_pedestrians[t].split('_')[3][0]]
                    self.targets_ID[t] = ped_class
            self.targets_t1 = np.copy(self.targets)
            for r in range(nrobots):
                self.robots[r][0:2] = np.copy(positions[r + ntargets])
                self.robots[r][2] = np.random.uniform(0, 360)
                if self.realistic_Dengine=="dummy_env":
                    self.drone_engines[r].resetFromPos(np.array([self.robots[r][0:2]]))
                if self.realistic_Dengine == "airsim":
                    set_pose_drone(self.clients[r], self.dronelist[r],
                                   [self.robots[r][0],self.robots[r][1],self.z_ref_drone[r]],
                                   [0,0,0],self.offset_envCenter, degrees=True)
                    # [0, 0, self.robots[r][2]], self.offset_envCenter, degrees = True)
                    self.z_drone_curr[r] = self.z_ref_drone[r]

            if self.save_scn:
                self.batch_scenario()
                if self.nepisode == self.max_episodes:
                    self.save_scenarios()
        else:
            assert self.dimensions==3
            self.load_scenario(random_targetID=True)
            self.targets_t1 = np.copy(self.targets)
            if self.env_mode == 'airsim':
                assert self.ntargets == len(self.name_pedestrians)
                # set initialised target poses in airsim
                for t in range(self.ntargets):
                    if replace_targets:
                        set_pose(self.client_ref, self.name_pedestrians[t],
                                 [self.targets[t][0], self.targets[t][1], self.z_ref_ped], [0, 0, self.targets[2]])
                    self.airsim_timestamp_t1.append(time.time())
                    # associate core targetIDs to airsim textures
                    ped_class = self.ped_class_dict[self.name_pedestrians[t].split('_')[2].split(self.pedestrian_model)[1][0]]
                    self.targets_ID[t] = ped_class # TODO: relate to the name of the pedestrian
            if self.realistic_Dengine == "airsim":
                for r in range(self.nrobots):
                    set_pose_drone(self.clients[r], self.dronelist[r],
                                   [self.robots[r][0], self.robots[r][1], self.z_ref_drone[r]],
                                   [0, 0, 0], self.offset_envCenter, degrees=True)
                    # [0, 0, self.robots[r][2]], self.offset_envCenter, degrees = True)
                    self.z_drone_curr[r] = self.z_ref_drone[r]

    # FUNCTION FOR GYM AND RL

    #def step(self, action, envtest=False): # TODO NOT NOW: turn logs off if not specified (less RAM consumed)
    def step(self, action_dict, envtest=False): # RLlib adaptation
        #print(time.time()-self.time_aux)
        """
        Takes the action from the robot and runs a step of the environment.
        It returns the next observation, the current reward and the terminal signal.
        """
        time_monitor = time.time()
        #print("step",self.steps)
        ## RLlib adaptation
        high_level_policy_update = self.steps % self.lh_ratio == 0
        if high_level_policy_update:
            action = list(action_dict.values())
            self.last_action = action.copy()
            # print("HIGH LEVEL POLICY!!!",action)
        else:
            action = self.last_action.copy()

        # print("start",time.time()-time_monitor)
        # Update timestep indicator
        self.steps += 1
        if high_level_policy_update:
            self.hl_steps +=1

        # Decide if we should update the belief
        high_level_belief_update = self.steps % self.lh_ratio == 0

        if self.env_mode!="airsim":
            # Save previous target velocity, used for target dynamics computation
            self.old_targets_vel = np.copy(self.targets_vel)
            # Move the targets
            if not envtest:
                self.move_targets(self.delta_t, dyn_sigma=self.dyn_sigma)  # move targets according to dynamic model "self.env_mode"

        else:
            # print("start.1",time.time()-time_monitor)
            self.targets, self.airism_timestamp = get_pedestrian_poses(self.client_ref, self.name_pedestrians, self.offset_envCenter)
            # print("start.2", time.time() - time_monitor)
            self.targets_vel = estimate_vel(self.targets,self.targets_t1,self.airism_timestamp,self.airsim_timestamp_t1)
            self.targets_t1 = np.copy(self.targets)
            self.airsim_timestamp_t1 = np.copy(self.airism_timestamp)
            if self.is_plotting_scene:
                for t in range(self.ntargets):
                    ce = (self.targets[t][0], self.targets[t][1])
                    self.targets_plots[t].shirt.set_center(ce)
                    self.targets_plots[t].shirt.set_theta1(-90 + self.targets[t][2])
                    self.targets_plots[t].shirt.set_theta2(90 + self.targets[t][2])
                    self.targets_plots[t].face.set_center(ce)
                    self.targets_plots[t].face.set_theta1(90 + self.targets[t][2])
                    self.targets_plots[t].face.set_theta2(270 + self.targets[t][2])
                    self.targets_plots[t].target_id.set_position(
                        (self.targets[t][0] + 0.5, self.targets[t][1] + 0.5, str(t)))

        last_location = np.copy(self.robots)

        # print("1", time.time() - time_monitor)
        # We apply the action to robot 0, moving it to its current chosen location in the neighborhood if possible
        for r in range(self.nrobots):
            # map actions to actual action space.
            # Map the action space to the circle of unit radius
            #if np.linalg.norm(action[r]) > 1: action[r] = action[r] / np.linalg.norm(action[r])
            # map actions to actual action space.
            action[r] = action[r] * self.action_space_mapping

            location = np.copy(self.robots[r])

            if self.realistic_Dengine != "dummy_env":
                if self.env_mode == 'cte_vel' or self.env_mode=='sf_goal' or self.env_mode=='brown' or self.env_mode=='static':
                    #applied_action = np.clip(action, self.action_space.low*self.delta_t, self.action_space.high*self.delta_t)
                    applied_action=action[r]*self.delta_t
                else:
                    applied_action = action[r]

                applied_rot = applied_action[2]
                location[0:2] += self.rotate_action_pos(applied_action[0:2],location[2]) #action[0:2]
                location[2] = (location[2] + 60 * applied_action[2]) % 360

            elif self.realistic_Dengine=="dummy_env": # INSERT REAL DYNAMICS
                action_global_frame_2d = self.rotate_action_pos(action[r][0:2],location[2])
                self.drone_engines[r].step(np.array([action_global_frame_2d]))

                applied_rot = action[r][2] * self.delta_t
                location[0:2] = self.drone_engines[r]._getDronePos()[0]
                location[2] = (location[2] + 60 * applied_rot) % 360


            if self.realistic_Dengine != "airsim": # DO AS NORMAL
                self.place_robot(r, location)
                self.last_action[r][0:2] = self.rotate_action_pos(self.last_action[r][0:2],-applied_rot*60)
            else:
                # APPLY ACTION IN AIRSIM
                action_global_frame_2d = self.rotate_action_pos(action[r][0:2], self.robots[r][2])
                viewpoint_global_pos = self.robots[r][0:2] + action_global_frame_2d  # * self.delta_t # multiplication only if teleporting

                viewpoint_global_yaw = (self.robots[r][2] + 60 * self.delta_t * action[r][2]) % 360

                # print("global yaw",location[2])
                # print("applied action", 60*action[r][2])
                # print("global viewpoint", viewpoint_global_yaw)
                # if time.time()-self.clock_last_action < self.high_level_delta_t:
                #     time.sleep(self.high_level_delta_t-(time.time()-self.clock_last_action))
                # print("Actual time:", time.time()-self.clock_last_action)
                self.clock_last_action = time.time()
                # position control
                move_drone2Pos(self.clients[r],self.dronelist[r],viewpoint_global_pos,self.offset_envCenter,self.z_ref_drone[r])
                # teleportation control
                # set_pose_drone(self.clients[r],self.dronelist[r],[viewpoint_global_pos[0],viewpoint_global_pos[1],self.z_ref_drone[r]],[0,0,0],self.offset_envCenter,degrees=True)
                # velocity control
                # move_drone2PosByVel(self.clients[r], self.dronelist[r], action_global_frame_2d, action[r][2]*60, self.z_ref_drone)
                move_camera2Orientation(self.clients[r], self.dronelist[r], viewpoint_global_yaw)

                # CHECK ROBOT POSE FROM AIRSIM
                self.robots[r], self.z_drone_curr[r] = get_2ddrone_pose(self.clients[r],self.dronelist[r], self.offset_envCenter)
                self.robots[r][2] = viewpoint_global_yaw


                if self.is_plotting_scene:
                    c, s = np.cos(np.radians(self.robots[r][2])), np.sin(np.radians(self.robots[r][2]))
                    t = np.array([0.5, 0.4])
                    R = np.array(((c, -s), (s, c)))
                    shift = R.dot(t)

                    ce = (self.robots[r][0], self.robots[r][1])
                    self.robots_plots[r].shape.set_xy(ce - shift)
                    self.robots_plots[r].shape.angle = self.robots[r][2]
                    self.robots_plots[r].fov.set_center(ce)
                    self.robots_plots[r].fov.set_theta1(self.robots[r][2] - self.robots_fov[r])
                    self.robots_plots[r].fov.set_theta2(self.robots[r][2] + self.robots_fov[r])

        # print("2", time.time() - time_monitor)
        # Now all positions from robots and targets are updated. Next: obtain new observation
        # Initialize entropy measurements (for observation and rewards) and logging variables
        num_targets_tracked = 0
        entropy_measurements = 0
        entropy_beliefs = 0
        new_targets_tracked = 0

        # Check visibility of targets, reused when obtaining probability estimates on target classes
        # print("2.1.1", time.time() - time_monitor)
        self.visibility = self.check_visibility() #TODO: needs to be updated to retrieve images from airsim
        # print("2.1.2", time.time() - time_monitor)
        self.detected_pedestrian_dict_list = [detectedPedestriansImg(self.clients[r], self.config, r) for r in range(self.nrobots)]
        # print("2.1.3", time.time() - time_monitor)
        # We obtain probability estimates and update beliefs on target class for all the targets
        for t in range(self.ntargets):
            new_belief = self.beliefs[0][t, :].copy()
            for r in range(self.nrobots):
                # We take probability estimates of the target t from the new location
                #if high_level_belief_update:
                # print("2.1", time.time() - time_monitor)
                self._observation[t, r] = self.observation(t, r) ## TODO: needs to be updated to retrieve images from airsim
                # print("2.2", time.time() - time_monitor)

                # If the belief is not fully certain, update using new observation
                if np.max(self.beliefs[0][t, :]) < BELIEF_THRESHOLD:
                    # Current probability estimate
                    prob_semantic = self._observation[t, r].copy()

                    # Compute entropies of the prob. estimate (mostly for observation and logs)
                    if not self.reward_1target or t == 0:
                        entropy_measurements += distrib.entropy(prob_semantic, base=2)

                    # Conflation of the two probabilities
                    #new_belief = self.beliefs[0][t, :] * prob_semantic / (np.dot(self.beliefs[0][t, :], prob_semantic)+1e-32)
                    #for _ in range(2): new_belief = new_belief * prob_semantic / (np.dot(new_belief, prob_semantic) + 1e-32)
                    if high_level_belief_update:
                        # print("Step: %2d, BELIEF UPDATE!!!" % (self.steps))
                        new_belief = new_belief * prob_semantic / (np.dot(new_belief, prob_semantic) + 1e-32)

            # Compute entropies of the belief (mostly for observation and logs)
            if not self.reward_1target or t == 0:
                entropy_beliefs += distrib.entropy(new_belief, base=2)

            # save last entropy (mostly for reward computation)
            # TODO NOT NOW: this can go outside from the for-loop
            self.last_entropy_log = entropy_beliefs

            # Update belief on target t for robot 0
            # TODO: DONE update to multirobot
            self.beliefs[0][t] = new_belief

            # Keep track of tracked targets by robot 0.  If centralized, this is not needed to be updated
            if not self.reward_1target or t==0:
                if np.max(self.beliefs[0][t, :]) >= 0.95:
                    if not self.num_targets_tracked[0][t]:
                        self.num_targets_tracked[0][t] = True
                        if self.beliefs[0][t][self.targets_ID[t]] >= 0.95:
                            self.targets_correct_classification[0][t] = 1
                        else:
                            self.targets_correct_classification[0][t] = -1
                        new_targets_tracked += 1
                    num_targets_tracked += 1

        #print("visibility:", self.visibility)
        #print("new observation", self._observation)

        # print("3", time.time() - time_monitor)
        # Reward computation. TODO: NOT NOW this could go in its own function
        # Initialization of the current reward
        reward = 0

        for r in range(self.nrobots):
            # Movement reward (punishment)
            # reward += -0.01 * np.linalg.norm(action[r][0:2])
            # reward += -0.01*np.linalg.norm(action[r][2])
            reward += -0.01 * np.linalg.norm(self.robots[r][0:2]-last_location[r][0:2])
            reward += -0.01 * np.linalg.norm((self.robots[r][2]-last_location[r][2])/60.0)

        # Reward for newly classified targets (centralized)
        reward += new_targets_tracked*5

        # Signal that identifies when an episode has been ended, either because the maximum number of timesteps
        # has been reached(horizon), all targets have been classified or the objective target has been identified(task).
        done_episode = (num_targets_tracked == self.ntargets) or (self.MAX_STEPS == self.steps) or (self.reward_1target and num_targets_tracked==1)

        # Signal that identifies when an episode has been ended due to task completion
        # No done is given if timeout has been hit. This is done to improve action value estimates. Only when the task
        # has been finished we use the actual reward value. Otherwise we bootstrap to be able to use small episodes
        # to learn a policy that generalizes to large episodes.
        done = (num_targets_tracked == self.ntargets) or (self.reward_1target and num_targets_tracked == 1)

        # During testing, timeout termination is needed to trigger episode termination
        if self.test and done_episode:
            done = True

        # Upon episode termination, compute final timestep reward (task completion) if necessary, and log if necessary.
        # ELSE: punish with timestep-related reward, and reward for the decrease in entropy in the beliefs.
        # TODO: (DONE) the entropy reward does not need to be updated if the belief updates are centralized
        if done:
            # Reward at the end of the episode: +20 if success, 0 if not id
            if self.MAX_STEPS > self.steps:
                reward += 100 #20 # MODIFIED
                self.success = 1
                # print(f'Success')
            else:
                reward += 0
            if self.logs:
                #self.episode_reward += reward
                self.log_performance_episode()
        else:
            # Constant penalty for processing a new measurement
            if high_level_belief_update: reward += -0.3
            # Variable penalty based on the entropy of the measurement (dense reward)
            if not math.isnan(entropy_beliefs) and len(self.episode_entropy) > 0:
                reward += self.episode_entropy[-1] - entropy_beliefs
                #print("entropy reward:", self.episode_entropy[-1] - entropy_beliefs)


        # print("4", time.time() - time_monitor)
        # Compute the observations given to the policy. TODO: (DONE) it has to return a list/dict and be adapted to multiple robots.
        obs = self._obs()

        # print("5", time.time() - time_monitor)
        # Logging beliefs and visibility
        self.episode_beliefs.append(np.copy(self.beliefs))
        self.episode_visibility.append(np.copy(self.visibility))

        # Updating the entropy over the beliefs of all targets.
        entropy = 0
        for t in range(self.ntargets):
            if not self.reward_1target or t == 0:
                entropy += distrib.entropy(self.beliefs[0][t, :], base=2)
                # if (self.beliefs[0][t] > 0.0) and (self.beliefs[0][t] < 1.0):
                #     entropy += self.beliefs[0][t]*np.log2(self.beliefs[0][t]) + (1-self.beliefs[0][t])*np.log2(1-self.beliefs[0][t])

        # Save to Log the entropy
        self.episode_entropy.append(entropy)

        # Save to Log poses of robots and targets
        poses = np.concatenate((np.copy(self.targets), np.copy(self.robots)))
        self.episode_poses.append(poses)

        # Only if render is active: update the plots on the beliefs
        if self.is_plotting_beliefs:
            for t in range(self.ntargets):
                for c in range(self.number_of_classes):
                    self.bar_plots[t][c].set_height(self.beliefs[self.robot_plot_belief][t, c])

        # return obs, reward, termination signal. {} can contain additional info.
        #return obs, reward, done, {}
        self.episode_reward += reward
        #RLlib adaptation
        obs_dict = self._make_dict(obs)
        reward_list = [np.copy(reward) for _ in range(self.nrobots)]
        rew_dict = self._make_dict(reward_list)
        done_list = [np.copy(done) for _ in range(self.nrobots)]
        done_dict = self._make_dict(done_list)
        done_dict["__all__"] = all(done_list)

        info_list = {'n':[{} for _ in range(self.nrobots)]}
        info_dict = self._make_dict([{"done": isdone} for isdone in done_list])
        for i, k in enumerate(info_dict):
            info_dict[k].update(info_list['n'][i])

        # self.time_aux = time.time()
        # if high_level_belief_update:
        #     print("HIGH LEVEL STEP:",self.hl_steps)
        # print("OBSERVATION:",obs_dict)
        # print("REWARD:", reward)
        # print("    Movement:", -0.01 * np.linalg.norm(self.robots[0][0:2]-last_location[0][0:2]) - 0.01 * np.linalg.norm((self.robots[0][2]-last_location[0][2])/60.0))
        # print("    New Targets:", new_targets_tracked*5)
        # if self.steps>1: print("    Entropy:", self.episode_entropy[-2] - entropy_beliefs)
        # print("    Time:",-0.3) if high_level_belief_update  else print("    Time:",0)
        # print("EPISODE REWARD:", self.episode_reward)

        # print("end", time.time() - time_monitor)
        return obs_dict, rew_dict, done_dict, info_dict

    def _obs(self):
        """
        Generate an observation over the environment. TODO: DONE update to multirobot
        """
        # Auxiliar variables used for logging and plotting
        obsed_targets = 0
        obsed_targets_tracked = 0
        tracked = 0
        # The observation for each robot depends on the number of targets in the scene TODO: DONE update to multirobots
        obs = []
        #robot_obs = {"targets":{"unidentified":[],"identified":[], "all":[]}}



        for r in range(self.nrobots): # compute observation for each robot
            robot_obs = {"targets": {"unidentified": [], "identified": [], "all": []}}
            # target observations TODO adapt observations and RUN the policy
            target_obs_identified = []
            target_obs_unidentified = []
            target_obs = []
            for t in range(self.ntargets):  # range(self.nrobots):
                aux = self.target_space.sample()
                loc = self.relative_pose(t, r)
                #loc = self.bounded_relative_pose(t, r) ## TRYING FOR SCALABILITY DURING TESTING
                aux['location'][0:2] = np.float32([loc[0:2]])

                aux['location'][2] = np.radians(np.float32([loc[2]]))
                rel_vel = self.relative_vel(t,r)
                aux['velocity'] = np.float32([rel_vel])[0]
                aux["belief"] = np.float32([1.0-distrib.entropy(self.beliefs[0][t, :], base=2)/distrib.entropy([1/self.number_of_classes]*self.number_of_classes, base=2)+1e-10])
                distrib.entropy(self.beliefs[0][t, :], base=2)
                aux["measurement"] = np.float32([1.0-distrib.entropy(self._observation[t, r], base=2)/distrib.entropy([1/self.number_of_classes]*self.number_of_classes, base=2)+1e-10])
                aux["tracked"] = np.float32([np.any(self.beliefs[0][t,:] > 0.95)])
                # print(obs)
                #target_obs.append(aux)
                if aux["tracked"] == 1:
                    target_obs_identified.append(aux)
                else:
                    target_obs_unidentified.append(aux)
                target_obs.append(aux)

                ### for logging
                if self.visibility_hist:
                    if aux["tracked"][0]:
                        tracked += 1
                    if aux["measurement"][0] > 2e-10:
                        obsed_targets += 1
                        if aux["tracked"][0]:
                            obsed_targets_tracked +=1

            # This part is for special cases when we need to order targets according to a certain metric
            # This is needed for baselines 1 and 2, and also for the LSTM ablation
            if self.heuristic_target_order: # change to general change
                if self.reverse_heuristic_target_order:
                    self.targets_id_order[r], aux_ = list(zip(*sorted(enumerate(target_obs), key=lambda x: (np.bool(x[1]['tracked']), np.linalg.norm(x[1]['location'][0:2])), reverse=True)))
                    target_obs = list(aux_)

                elif self.robot_target_assignment:
                    # print(self.targets_id_order)
                    if self.targets_id_order[r] == [] or target_obs[self.targets_id_order[r][0]]["tracked"]:
                        self.targets_id_order[r], aux_ = list(zip(*sorted(enumerate(zip(target_obs, self.target_assigned)), key=lambda x: (
                                np.bool(x[1][0]['tracked']), np.bool(x[1][1]),
                                np.linalg.norm(x[1][0]['location'][0:2])))))
                    target_obs = [target_obs.pop(self.targets_id_order[r][0])] + target_obs
                    self.target_assigned[self.targets_id_order[r][0]] = True

                else:
                    if self.targets_id_order[r] == [] or target_obs[self.targets_id_order[r][0]]["tracked"]:
                        self.targets_id_order[r], aux_ = list(zip(*sorted(enumerate(target_obs), key=lambda x: (np.bool(x[1]['tracked']), np.linalg.norm(x[1]['location'][0:2])))))
                    target_obs = [target_obs.pop(self.targets_id_order[r][0])] + target_obs

            # Only for baseline 1: cropping the observation to only the first target
            if self.heuristic_policy:
                for t in range(self.ntargets):
                    if not target_obs[t]['tracked']:
                        target_obs = [target_obs[t]]
                        break


            robot_obs["targets"]["unidentified"] = target_obs_unidentified
            robot_obs["targets"]["identified"] = target_obs_identified
            robot_obs["targets"]["all"] = target_obs

            # observations on the locations of other robots
            if self.multiagent_policy:
                other_robot_obs = []
                for r_other in range(self.nrobots):
                    if r_other == r: continue
                    aux_robot = self.robot_space.sample()
                    loc_robot = self.relative_pose_rXr(r_other, r)
                    aux_robot['location'][0:2] = np.float32([loc_robot[0:2]])
                    aux_robot['location'][2] = np.radians(np.float32([loc_robot[2]]))
                    other_robot_obs.append(aux_robot)
                robot_obs["robots"] = other_robot_obs
                #obs.append([target_obs,other_robot_obs])
            #else:
            #    obs.append(target_obs)

            obs.append(robot_obs)
        # For logging purposes: simultaneous observations (total and tracked) + total targets tracked
        if self.visibility_hist:
            self.obs_histogram.append(obsed_targets)
            self.obs_tracked_histogram.append(obsed_targets_tracked)
            self.tracked_histogram.append(tracked)

        # if self.steps==0 or self.steps % self.lh_ratio == self.lh_ratio-1 or self.lh_ratio == 1 or True:
        #     print("step: %2d, OBSERVATION:" % (self.steps), obs)

        if self.deglossed_obs: return obs
        else: return [obs[i]["targets"]["all"] for i in range(self.nrobots)] # apparently this works better in our old policies TODO


    def observation(self, target, robot):  # TODO make the yolo version faster (only classify detectable targets according to heuristics)
        if self.simulated_perception == "dummy":
            prob = np.ones(self.number_of_classes) / self.number_of_classes
            if self.visibility[robot, target]:
                if self.dimensions == 3: # ASK EDUARDO
                    # SCENE3D observation depends on the image projection
                    # TODO NOT NOW for the moment this only works with two classes
                    pol = self.get_projection(target, robot)
                    # print(f'Area[{t}] = {pol.area}')
                    x, y = pol.exterior.xy

                    boundaries = (np.min(x) > 0) and (np.max(x) < 640) and (np.min(y) > 0) and (np.max(y) < 480)
                    width = np.abs(x[1] - x[0])
                    height = np.max([np.abs(y[2] - y[1]), np.abs(y[3] - y[0])])
                    area = width * height
                    skew = np.abs(y[1] - y[0])
                    AMIN = 40000
                    if boundaries:
                        if area > AMIN:
                            aux = 0.5 * np.exp(-skew / 20.0)
                        else:
                            aux = 0.5 * np.exp(-skew / 20.0) * area / AMIN
                    else:
                        aux = 0.05 * np.exp(-skew / 20.0)

                    prob = np.ones(self.number_of_classes) * (1 - aux - 1 / self.number_of_classes) / (
                            self.number_of_classes - 1)
                    prob[self.targets_ID[target]] = aux + 1 / self.number_of_classes

                    if self.is_plotting_images:
                        self.images[robot][target].xy = np.copy(np.array([x, y]).transpose())
                        self.images[robot][target].set_visible(True)

                if self.is_plotting_scene: #and (robot == 0):
                    self.targets_plots[target].face.fill = False

            else:  # TARGET IS NOT VISIBLE (only update the plots)
                if (self.dimensions == 3) and self.is_plotting_images:
                    self.images[robot][target].set_visible(False)
                if self.is_plotting_scene and (robot == 0):
                    self.targets_plots[target].face.fill = True

            if self.num_targets_tracked[0][target] and self.render_enabled:
                self.targets_plots[target].shirt.set_facecolor((0.031249343749343732, 1.0, 1.3125013125390552e-06, 1.0))#shirt_color = (0.031249343749343732, 1.0, 1.3125013125390552e-06, 1.0)
            elif len(self.targets_plots)!=0:
                self.targets_plots[target].shirt.set_facecolor(self.colors(self.targets_ID[target]))
        elif self.simulated_perception == "simulated":
            prob = np.ones(self.number_of_classes) / self.number_of_classes
            if self.visibility[robot, target]:
                rel_pose = self.relative_pose(target,robot)
                rel_pose[0:2] = np.float32([rel_pose[0:2]])
                rel_pose[2] = np.radians(np.float32([rel_pose[2]]))
                prob = self.unrealPedestrianData[self.targets_ID[target]].getProb(rel_pose)

                if rel_pose[0] > self.unrealPedestrianData[self.targets_ID[target]].max_x or \
                    rel_pose[0] < 0.5 or \
                    rel_pose[1] < self.unrealPedestrianData[self.targets_ID[target]].min_y or \
                    rel_pose[1] > self.unrealPedestrianData[self.targets_ID[target]].max_y:
                        prob = np.ones_like(prob)/self.number_of_classes

            if not np.array_equal(prob, np.ones_like(prob)/self.number_of_classes): #aka information is visible
                if self.is_plotting_images:
                    pol = self.get_projection(target, robot)
                    # print(f'Area[{t}] = {pol.area}')
                    x, y = pol.exterior.xy
                    self.images[robot][target].xy = np.copy(np.array([x, y]).transpose())
                    self.images[robot][target].set_visible(True)
                if self.is_plotting_scene:  # and (robot == 0):
                    self.targets_plots[target].face.fill = False
            else:  # TARGET IS NOT VISIBLE (only update the plots)
                if (self.dimensions == 3) and self.is_plotting_images:
                    self.images[robot][target].set_visible(False)
                if self.is_plotting_scene and (robot == 0):
                    self.targets_plots[target].face.fill = True
            if self.num_targets_tracked[0][target] and self.render_enabled:
                self.targets_plots[target].shirt.set_facecolor((0.031249343749343732, 1.0, 1.3125013125390552e-06, 1.0))#shirt_color = (0.031249343749343732, 1.0, 1.3125013125390552e-06, 1.0)
            elif len(self.targets_plots)!=0:
                self.targets_plots[target].shirt.set_facecolor(self.colors(self.targets_ID[target]))

        elif self.simulated_perception == "simulated_airsim":
            prob = np.ones(self.number_of_classes) / self.number_of_classes
            if self.visibility[robot, target]:
                rel_pose = self.relative_pose(target,robot)
                rel_pose[0:2] = np.float32([rel_pose[0:2]])
                rel_pose[2] = np.radians(np.float32([rel_pose[2]]))
                # prob = self.unrealPedestrianData[self.targets_ID[target]].getProb(rel_pose)
                prob = self.unrealPedestrianData[self.targets_ID[target]].getProb_yolo(rel_pose, self.yolo)

                if rel_pose[0] > self.unrealPedestrianData[self.targets_ID[target]].max_x or \
                    rel_pose[0] < 0.5 or \
                    rel_pose[1] < self.unrealPedestrianData[self.targets_ID[target]].min_y or \
                    rel_pose[1] > self.unrealPedestrianData[self.targets_ID[target]].max_y:
                        prob = np.ones_like(prob)/self.number_of_classes

            if not np.array_equal(prob, np.ones_like(prob)/self.number_of_classes): #aka information is visible
                if self.is_plotting_images:
                    pol = self.get_projection(target, robot)
                    # print(f'Area[{t}] = {pol.area}')
                    x, y = pol.exterior.xy
                    self.images[robot][target].xy = np.copy(np.array([x, y]).transpose())
                    self.images[robot][target].set_visible(True)
                if self.is_plotting_scene:  # and (robot == 0):
                    self.targets_plots[target].face.fill = False
            else:  # TARGET IS NOT VISIBLE (only update the plots)
                if (self.dimensions == 3) and self.is_plotting_images:
                    self.images[robot][target].set_visible(False)
                if self.is_plotting_scene and (robot == 0):
                    self.targets_plots[target].face.fill = True
            if self.num_targets_tracked[0][target] and self.render_enabled:
                self.targets_plots[target].shirt.set_facecolor((0.031249343749343732, 1.0, 1.3125013125390552e-06, 1.0))#shirt_color = (0.031249343749343732, 1.0, 1.3125013125390552e-06, 1.0)
            elif len(self.targets_plots)!=0:
                self.targets_plots[target].shirt.set_facecolor(self.colors(self.targets_ID[target]))

        elif self.simulated_perception == "airsim":
            prob = np.ones(self.number_of_classes) / self.number_of_classes
            if self.visibility[robot, target]:
                rel_pose = self.relative_pose(target,robot)
                rel_pose[0:2] = np.float32([rel_pose[0:2]])
                rel_pose[2] = np.radians(np.float32([rel_pose[2]]))
                # prob = self.unrealPedestrianData[self.targets_ID[target]].getProb(rel_pose)
                #prob = self.unrealPedestrianData[self.targets_ID[target]].getProb_yolo(rel_pose, self.yolo)
                if self.name_pedestrians[target] in self.detected_pedestrian_dict_list[robot].keys():
                    image = self.detected_pedestrian_dict_list[robot][self.name_pedestrians[target]]
                    # prob = self.unrealPedestrianData[self.targets_ID[target]].getProbFromImg_yolo(rel_pose,image,self.yolo)
                    ttaux = time.time()
                    prob = self.unrealPedestrianData[self.targets_ID[target]].getProbFromImg_yolo(rel_pose, image,self.yolo,oracle=False)
                    # print(time.time()-ttaux,"time for target ",target,"with rel. pose",rel_pose)
                else:
                    prob = np.ones_like(prob) / self.number_of_classes

            if not np.array_equal(prob, np.ones_like(prob)/self.number_of_classes): #aka information is visible
                if self.is_plotting_images:
                    pol = self.get_projection(target, robot)
                    # print(f'Area[{t}] = {pol.area}')
                    x, y = pol.exterior.xy
                    self.images[robot][target].xy = np.copy(np.array([x, y]).transpose())
                    self.images[robot][target].set_visible(True)
                if self.is_plotting_scene:  # and (robot == 0):
                    self.targets_plots[target].face.fill = False
            else:  # TARGET IS NOT VISIBLE (only update the plots)
                if (self.dimensions == 3) and self.is_plotting_images:
                    self.images[robot][target].set_visible(False)
                if self.is_plotting_scene and (robot == 0):
                    self.targets_plots[target].face.fill = True
            if self.num_targets_tracked[0][target] and self.render_enabled:
                self.targets_plots[target].shirt.set_facecolor((0.031249343749343732, 1.0, 1.3125013125390552e-06, 1.0))#shirt_color = (0.031249343749343732, 1.0, 1.3125013125390552e-06, 1.0)
            elif len(self.targets_plots)!=0:
                self.targets_plots[target].shirt.set_facecolor(self.colors(self.targets_ID[target]))

        return prob

    def log_performance_episode(self):
        """
        Function to generate a csv file and save the performance parameters along an episode of the current policy.
        """
        foldername = self.log_folder + '/test_performance'
        if self.heuristic_policy:
            foldername = foldername + '_heuristic'
        if self.heuristic_target_order:
            foldername = foldername + '_heuristicTargetOrder'
        if self.MAX_STEPS != 100:
            foldername = foldername + '_' + str(int(self.MAX_STEPS/self.lh_ratio))
        if self.visibility_hist:
            foldername = foldername + '_histogram'
        if self.robot_robot_occlusion:
            foldername = foldername + '_robrobOcclusion'
        if self.robot_target_assignment:
            foldername = foldername + '_rotarAssign'
        if SIDE != 8.0:
            if SIDE==None:
                foldername = foldername + '_airSimEnv'
            else:
                foldername = foldername + '_' + str(SIDE)
        if self.realistic_Dengine=="dummy_env":
            foldername = foldername + '_realistic'
        elif self.realistic_Dengine=="airsim":
            foldername = foldername + '_airsim'
        if self.lh_ratio!=1:
            foldername = foldername + '_' + str(self.lh_ratio)
        if self.number_of_classes!=2:
            foldername = foldername + '_' + str(self.number_of_classes)+'classes'
        if self.dyn_sigma!=0:
            foldername = foldername + '_dynSigma' + str(self.dyn_sigma)
        if self.simulated_perception != "dummy":
            foldername = foldername + '_per_' + str(self.simulated_perception)
        if self.dronelist[0].vel < 3.0:
            foldername = foldername + '_slowmo'


        foldername = foldername + '_'+self.env_mode
        # foldername = foldername + '.csv'
        # foldername = foldername + '_newtrial.csv'
        foldername = foldername + '_video.csv'
        #foldername = foldername + 'twice_observed.csv'

        csvDict = OrderedDict({
            'episode': self.nepisode,
            'nrobots': self.nrobots,
            'ntargets': self.ntargets,
            'nsteps': self.steps,
            'success': self.success,
            'nb_targets_tracked': np.sum(self.num_targets_tracked[0]),
            'entropy': self.last_entropy_log,
            'reward': self.episode_reward,
            'nb_targets_missclassified': -np.sum(self.targets_correct_classification[0][self.targets_correct_classification[0] == -1])
        })

        if self.visibility_hist:
            csvDict['obs_targets'] = ','.join(str(e) for e in self.obs_histogram)
            csvDict['obs_tracked_targets'] = ','.join(str(e) for e in self.obs_tracked_histogram)
            csvDict['tracked_targets'] = ','.join(str(e) for e in self.tracked_histogram)


        trainEvlogs2csv(foldername, csvDict, csvDict['episode'])

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen. If established a priori, saves the generated images.
        """
        # print current step and beliefs
        print(f'Step: {self.steps}')
        # print(f'Belief: {self.beliefs}')

        # Creates the folder where the images and videos are respectively stored and generated.
        if self.save_video: self.video_foldername = self.log_folder+'/videos/'+str(self.ntargets)+'_targets_episode_' + str(self.nepisode)+'/'
        if self.save_video and not os.path.isdir(self.video_foldername):
            os.makedirs(self.video_foldername+'camera/')
            os.makedirs(self.video_foldername + 'state/')
            os.makedirs(self.video_foldername + 'beliefs/')

        # Initialize the window used for rendering
        if not self.render_enabled:
            self.plot_scene = True
            self.plot_beliefs = True
            self.plot_image = False
            if self.plot_scene:
                self.fig, self.ax = self.init_plot_scene(num_figure=1, SIDEx=SIDEx, SIDEy=SIDEy)
                # camera = Camera(fig)
            if self.plot_beliefs:
                self.fig2, self.ax2 = self.init_plot_beliefs(num_figure=2)
            if self.plot_image:
                self.fig3, self.ax3 = self.init_plot_images(num_figure=3)

            if self.plot_image or self.plot_beliefs or self.plot_scene:
                # time.sleep(10.0)
                plt.show()
            self.render_enabled = True

        # Update the rendered plots
        else:
            # Update plots of the scene
            if self.plot_scene:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            # Update plot of the image
            if self.plot_image:
                self.fig3.canvas.draw()
                self.fig3.canvas.flush_events()
            # Update plots of beliefs
            if self.plot_beliefs:
                self.fig2.canvas.draw()
                self.fig2.canvas.flush_events()

        # Only render if we are testing the policy
        if self.test:
            if self.save_video and self.steps>1:
                if self.plot_image: self.fig3.savefig(self.video_foldername+'camera/'+str(self.steps)+'.png')
                if self.plot_scene: self.fig.savefig(self.video_foldername + 'state/' + str(self.steps) + '.png')
                if self.plot_beliefs: self.fig2.savefig(self.video_foldername + 'beliefs/' + str(self.steps) + '.png')

def quaternion_to_euler(x, y, z, w):

    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

def get_pedestrian_poses(client, name_pedestrians, offset):
    pedestrian_poses = [np.zeros(3) for name_ped in name_pedestrians]
    measure_timestamps = np.zeros(len(name_pedestrians))
    for i,name_ped in enumerate(name_pedestrians):
        # ped_pose = client.simGetObjectPoseAsync(name_ped)
        ped_pose = client.simGetObjectPose(name_ped)
        timestamp = time.time()
        pos_x = ped_pose.position.x_val + offset[0]
        pos_y = (-ped_pose.position.y_val + offset[1])
        _,_,yaw = quaternion_to_euler(ped_pose.orientation.x_val,
                                    ped_pose.orientation.y_val,
                                    ped_pose.orientation.z_val,
                                    ped_pose.orientation.w_val)
        pedestrian_poses[i] = np.array([pos_x,pos_y,-yaw])
        measure_timestamps[i] = timestamp
    return pedestrian_poses, measure_timestamps

def estimate_vel(measurement2, measurement1, timestamps2, timestamps1):
    assert len(measurement2)==len(measurement1)
    target_vel21 = [(measurement2[i][0:2] - measurement1[i][0:2]) / (timestamps2[i] - timestamps1[i]) for i in
                    range(len(measurement1))]
    return target_vel21

def set_pose(client, name, position, orientation, offset, degrees=True):  # position --> list(3) / orientation --> list(3)
    if degrees:
        orientation = [np.deg2rad(angle) for angle in orientation]
    pose = airsim.Pose(airsim.Vector3r(position[0]-offset[0], (-position[1]+offset[1]), position[2]),
                       airsim.to_quaternion(orientation[0], orientation[1], -orientation[2]))
    client.simSetObjectPose(name, pose, True)

def set_pose_drone(client, drone, position, orientation,offset, degrees=False):
    _, task = drone.moveTOpos(client, [position[0]-offset[0],(-position[1]+offset[1]),position[2]], -orientation[2])
    if degrees:
        orientation = [np.deg2rad(angle) for angle in orientation]
    # set orientation
    pose = airsim.Pose(airsim.Vector3r(position[0]-offset[0], (-position[1]+offset[1]), position[2]),
                       airsim.to_quaternion(orientation[0], orientation[1], -orientation[2]))
    client.simSetVehiclePose(pose, True, vehicle_name=drone.name)
    return client

def move_drone2Pos(client, drone, position2d, offset,zheight):
    #_, task = drone.moveTOpos(client, [position2d[0] - offset[0], (-position2d[1] + offset[1]), zheight],-yaw)
    _, task = drone.moveTOpos(client, [position2d[0] - offset[0], (-position2d[1] + offset[1]), zheight])
    return task

def move_drone2PosByVel(client, drone, vel2d, yaw, zheight):
    # print(zheight)
    # _, task = drone.moveTOpos(client, [position2d[0] - offset[0], (-position2d[1] + offset[1]), zheight],-yaw)
    # _, task = drone.moveByVel(client, [vel2d[0],-vel2d[1],zheight], yaw=-yaw, duration=0.25)
    # _, task = drone.moveByVelZ(client, [vel2d[0], -vel2d[1]], yaw=-yaw, duration=1.0, zheight=zheight)
    _, task = drone.moveByVelZ(client, [vel2d[0], -vel2d[1]], yaw=0, duration=1.0, zheight=zheight)
    return task

def move_camera2Orientation(client, drone, yaw):
    camera_pose =  airsim.Pose(airsim.Vector3r(0,0,0.2), airsim.to_quaternion(np.deg2rad(-20),0,np.deg2rad(-yaw)))
    client.simSetCameraPose("0", camera_pose, vehicle_name=drone.name)


def get_2ddrone_pose(client, drone, offset):
    state = drone.get_state(client)
    # print(state)
    pos_x = state.position.x_val + offset[0]
    pos_y = (-state.position.y_val + offset[1])
    pos_z = state.position.z_val
    _, _, yaw = quaternion_to_euler(state.orientation.x_val,
                                    state.orientation.y_val,
                                    state.orientation.z_val,
                                    state.orientation.w_val)
    return np.array([pos_x,pos_y,-yaw]), pos_z

def detectedPedestriansImg(client, config, i): #todo use to obtain rendered images
    images = getResponseImagesIdx([client], config, i)
    cameras_names = [config.camera_names[i]]
    gt2d_pedestrians = {}
    for cam_name in cameras_names:
        gt2d_pedestrians[cam_name] = {}
    frame_index = 0
    frame_index_key = str(frame_index)
    frame_index_key = frame_index_key.zfill(4)
    for cam in cameras_names:
        # Update 2d pedestrians bbox obtained
        gt2d_pedestrians[cam][frame_index_key] = []
        gt2d_pedestrians, _ = update_gt2d_pedestrian(client, cam, gt2d_pedestrians, frame_index_key, config)
    ped_images = visualize(cameras_names, images, frame_index, gt2d_pedestrians)
    return ped_images

def detectedPedestriansImg_4Plots(client, config, i): #todo use to obtain rendered images
    images = getResponseImagesIdx([client], config,i)
    cameras_names = [config.camera_names[i]]
    gt2d_pedestrians = {}
    for cam_name in cameras_names:
        gt2d_pedestrians[cam_name] = {}
    frame_index = 0
    frame_index_key = str(frame_index)
    frame_index_key = frame_index_key.zfill(4)
    for cam in cameras_names:
        # Update 2d pedestrians bbox obtained
        gt2d_pedestrians[cam][frame_index_key] = []
        gt2d_pedestrians, _ = update_gt2d_pedestrian(client, cam, gt2d_pedestrians, frame_index_key, config)
    ped_images, images_bbox = visualize_4Plots(cameras_names, images, frame_index, gt2d_pedestrians, print_bbox = False)
    return ped_images, images


if __name__ == "__main__":
    # """
    # Used for testing the environment
    # """
    # from utils.airsim.pedestrians import get_name_pedestrians, update_gt3d_pedestrian
    # from utils.airsim.settings_airsim import Configuration
    # import airsim
    # from utils.airsim.drone import MultiRotor
    # from utils.airsim.image_utils import getResponseImages, save_images
    #
    # # Defining settings
    # velocity = 2
    # save_mode = 'start'  # 'start', 'wait'
    # frames_to_capture = 100
    # name_experiment = 'ped3_class0'
    # img_types = 'RGB'  # 'RGB' , 'RGB-D', 'RGB-DS'
    # drone1 = MultiRotor("Drone1", [0, 0, 0], velocity)
    # dronelist = [drone1]
    #
    # config = Configuration(img_types, frames_to_capture, save_mode, visualize_images=True, vis_pedestrian_2dGT=True,name_experiment=name_experiment,external=False, uavs = dronelist)
    # # Initializing Vehicle of AirSim in Unreal
    # clients = [airsim.VehicleClient() for _ in range(config.number_cameras)]
    #
    # cameras_names = config.camera_names
    # gt3d_pedestrians, gt2d_pedestrians = {}, {}
    # print('-> External cameras defined {} in MODE: {} '.format(cameras_names, config.mode))
    # offset_envCenter = np.array([12.45, 10.6])
    # # -------------Connect AirSim with Unreal--------------
    # client_ref = clients[0]
    # client_ref.confirmConnection()
    #
    # struct_ref = 'BP_P_'  # reference structure in Unreal to look for the pedestrians
    # name_pedestrians = get_name_pedestrians(client_ref, struct_ref, segmentation=config.semantic_segmentation)
    # print('-> Pedestrians found in the scene: {}'.format(name_pedestrians))
    # print("Number of pedestrians:", len(name_pedestrians))
    #
    # # frame_index = 0
    # # frame_index_key = str(frame_index)
    # # frame_index_key = frame_index_key.zfill(4)
    # # images = getResponseImages(clients, config)
    #
    # ntests = 10
    # pedestrian_poses1, timestamps1 = get_pedestrian_poses(client_ref, name_pedestrians, offset_envCenter)
    # print(pedestrian_poses1)
    # time.sleep(0.05)
    #
    # for i in range(ntests):
    #     pedestrian_poses2, timestamps2 = get_pedestrian_poses(client_ref, name_pedestrians, offset_envCenter)
    #     target_vel21 = estimate_vel(pedestrian_poses2,pedestrian_poses1,timestamps2,timestamps1)
    #     print(target_vel21)
    #     pedestrian_poses1 = pedestrian_poses2
    #     timestamps1 = timestamps2
    #     time.sleep(0.2)
    #
    # ped_pose = client_ref.simGetObjectPose(name_pedestrians[0])
    # z_ref = ped_pose.position.z_val
    # print(z_ref)
    # set_pose(client_ref,name_pedestrians[0],[-offset_envCenter[0],-offset_envCenter[1],z_ref],[0,0,0], offset_envCenter)
    # pedestrian_poses1, timestamps1 = get_pedestrian_poses(client_ref, name_pedestrians, offset_envCenter)
    # print(pedestrian_poses1)

    # """
    ## Testing yolo
    from utils.yolo_model import create_trained_yolo
    yolo = create_trained_yolo()
    print("yolo created")
    from utils.approx_perception import test_function
    # test_function(real_class=1,nclasses=10)    
    
    config = {
        "dimensions": 3,
        "ntargets": [1],
        "nrobots": [1],
        "nclasses": 2,
        "MAXTARGETS":10,
        "horizon": 100,
        "env_mode": "airsim",
        "test": False,
        "heuristic_target_order": False,
        "reward_1target":False,
        "random_beliefs": False,
        "random_static_dynamic":True,
        "reverse_heuristic_target_order": False,
    }

    # Initialize plots stuff
    plt.close('all')
    plt.ion()

    print('Init Scene')
    sc = SceneEnv(config)
    sc.yolo = create_trained_yolo()
    obs=sc.reset(replace_targets = False)
    print(obs)
    # set_pose_drone(sc.client_ref,sc.drone_ref,[0,0,sc.z_ref_drone],[0,0,0],sc.offset_envCenter,degrees=True)
    # move_camera2Orientation(sc.client_ref, sc.drone_ref, 0)
    print("reset happened")
    # sc.robots[0] = np.array([0, 0, 0.0])
    # sc.robots[1] = np.array([0, 0.0, 180.0])
    # sc.targets[0] = np.array([4.0, 0.0, 0.0])
    # # sc.targets_ID[0] = 0
    # print(sc.targets_ID[0])
    #

    ## From Here this is testing
    plot_scene = True
    if plot_scene:
        fig, ax = sc.init_plot_scene(num_figure=1)
        # camera = Camera(fig)

    plot_beliefs = False
    if plot_beliefs:
        fig2, ax2 = sc.init_plot_beliefs(num_figure=2)

    plot_image = False
    if plot_image:
        fig3, ax3 = sc.init_plot_images(num_figure=3)

    if plot_image or plot_beliefs or plot_scene:
        plt.show()

    # # MAIN LOOP
    final = False
    taux = 1
    sc.render_enabled = False
    # we ran one full episode with random actions to check that everything works
    #"""
    while taux <= 10000:
        action = {'0':np.array([1., 0., 0]),
                  # '1':np.array([1., 0., 0]),
                  # '2':np.array([1., 0., 0])
                  }#,'1':np.array([0, 0, 0.25])}#np.random.uniform(-1, 1, 3)
        #action[0] = 0
        #action['0'][2] *= -1**taux
        time_aux = time.time()
        obs, reward, final, _ = sc.step(action) #, envtest = True)
        print(time.time()-time_aux)
        print("robots:", sc.robots)
        print("targets:",sc.targets)
        # print(time.time()-time_aux)
        print(obs)
        #print(sc.beliefs[0])
        #print(sc._observation)
        #print(reward)
        #print("class:", sc.targets_ID[0])
        #image, _, _ = sc.unrealPedestrianData[sc.targets_ID[0]].getObservation(sc.relative_pose(0,0))

        #plt.imshow(image)

        # Update plots of the scene
        if plot_scene:
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Update plot of the image
        if plot_image:
            fig3.canvas.draw()
            fig3.canvas.flush_events()

        # Update plots of beliefs
        if plot_beliefs:
            fig2.canvas.draw()
            fig2.canvas.flush_events()

        # if plot_image or plot_beliefs or plot_scene:
        #     time.sleep(0.1)

        if (taux % 100 == 0 and taux!=0) or all(final):
            sc.reset()
        taux +=1

    print('everything worked properly')


    #detected_peds, front_image = detectedPedestriansImg_4Plots(sc.client_ref, sc.config, 0)

    #plot figures FROM HERE
    # import cv2
    # from utils.approx_perception import segmentRedHSV, obtain_class_probabilities
    # from yolov3.utils import detect_number, detect_number_FOV
    # from yolov3.configs import *
    #
    # for robot in range(sc.nrobots):
    #     print("ROBOT",robot)
    #     detected_peds, front_image = detectedPedestriansImg_4Plots(sc.clients[robot], sc.config, robot)
    #     # plt.figure()
    #     ## Plots when considering all the FOV
    #     # for key in front_image.keys():
    #     #     front_image[key]['rgb'] = cv2.cvtColor(front_image[key]['rgb'], cv2.COLOR_BGR2RGB)
    #     # image_to_print = front_image['Drone'+str(robot+1)]['rgb']
    #     # plt.figure()
    #     # plt.imshow(image_to_print)
    #     #
    #     # image_to_print = segmentRedHSV(image_to_print)
    #     # plt.figure()
    #     # plt.imshow(image_to_print)
    #     #
    #     # bboxes, image_to_print = detect_number_FOV(yolo, image_to_print, "", input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES,
    #     #                    rectangle_colors=(255, 0, 0),show_image = True)# human_image=original_image)
    #     # plt.figure()
    #     # plt.imshow(image_to_print)
    #     #### END
    #     for pedkey in detected_peds.keys():
    #         # plt.figure()
    #         detped = cv2.cvtColor(detected_peds[pedkey], cv2.COLOR_BGR2RGB)
    #         # print(detped.shape)
    #         # orig_size_x = detped.shape[0]
    #         # orig_size_y = detped.shape[1]
    #         human_image = cv2.resize(detped, (416, 416))
    #         plt.figure()
    #         plt.imshow(cv2.cvtColor(human_image, cv2.COLOR_RGB2HSV))
    #         segmentedpic = segmentRedHSV(detected_peds[pedkey])
    #         plt.figure()
    #         plt.imshow(segmentedpic)
    #         plt.figure()
    #         residual = cv2.cvtColor(human_image, cv2.COLOR_RGB2HSV)
    #         residual[segmentedpic==0]=0
    #         plt.imshow(residual)
    #         if robot == 1:
    #             #plt.figure()
    #             probas1 = obtain_class_probabilities(segmentedpic, yolo, 10, human_image, show_image=True)
    #         else:
    #             probas1 = obtain_class_probabilities(segmentedpic, yolo, 10, human_image, show_image=False)
    #         print(probas1)
    # TO HERE
    # sc.plot_episode(num_figure=1, file_name='caca')
    #
    # name = 'aux'
    # im = sc.plot_video_episode(file_name=name, show_all_poses=False)
    #"""



