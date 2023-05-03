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
import gym
from gym import spaces
from ray import rllib
from ray.rllib.utils.spaces.repeated import Repeated
import matplotlib.gridspec as gridspec
from celluloid import Camera
import gc

import csv
from collections import OrderedDict

# import importlib
# importlib.reload(Scene)

import os
from utils.approx_perception import Pedestrian

# Constants (TODO NOT NOW to edit them from the main file)
MAXTARGETS = 8
MAX_STEPS = 100
SIDE = 25 #25 #25.0 #25 #8.0
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
class SceneEnv(rllib.MultiAgentEnv):
    def __init__(self, config):

        self.dimensions = 3  # its use is deprecated, only 3 dimensions (xpos, ypos, orientation) are considered.

        if "ntargets" in config:
            self.ntargets = config["ntargets"][-1] # records the current number of targets
            self.nmaxtargets = config["ntargets"]  # records the bounds on the number of targets (min,max)
            self.MAX_TARGETS = config["ntargets"][-1] # config["MAXTARGETS"] # max number of targets
            assert len(config["ntargets"])==1 or len(config["ntargets"])==2
            if len(config["ntargets"])==2:
                assert config["ntargets"][0] < config["ntargets"][1]
        else:
            self.ntargets = 1
            self.nmaxtargets = [1]
            self.MAX_TARGETS = 1

        if "nrobots" in config:
            self.nrobots = config["nrobots"][-1]  # currrent episode's number of robots
            self.nmaxrobots = config["nrobots"]  # bounds on the number of robots
            self.MAX_ROBOTS = config["nrobots"][-1]  # max number of robots
            assert len(config["nrobots"])==1 or len(config["nrobots"])==2
            if len(config["nrobots"])==2:
                assert config["nrobots"][0] < config["nrobots"][1]
        else:
            self.nrobots = 1
            self.nmaxrobots = [1]
            self.MAX_ROBOTS = 1

        if "nclasses" in config:
            self.number_of_classes = config["nclasses"]
        else:
            self.number_of_classes = 2

        # Position of targets
        self.targets = []

        # Label (true class) of each target
        self.targets_ID = []
        self.colors = auxFunctions.get_cmap(self.number_of_classes)

        if "test" in config:
            self.test = config["test"]
        else:
            self.test = False

        if "heuristic_policy" in config:
            self.heuristic_policy = config["heuristic_policy"]
        else:
            self.heuristic_policy = False

        if "max_episodes" in config:
            self.max_episodes = config["max_episodes"]
        else:
            self.max_episodes = None

        if self.test:
            if "save_scn" in config and config['save_scn']:
                self.save_scn = config["save_scn"]
                self.save_scn_folder = config["save_scn_folder"]
                self.scenario_saved_batch = []
            else:
                self.save_scn = False
        else:
            self.save_scn = False

        self.heuristic_target_order = config["heuristic_target_order"] or config["reverse_heuristic_target_order"]
        self.reverse_heuristic_target_order = config["reverse_heuristic_target_order"]
        if self.heuristic_target_order:
            self.targets_id_order = [[] for r in range(self.nrobots)]

        self.robot_target_assignment = True and self.heuristic_target_order

        self.num_agents = self.nrobots
        # Position of the robots
        self.robots = []
        self.robots_fov = []

        # Matrix of beliefs (nrobots x ntargets x nclasses)
        self.beliefs = []

        # Number of targets with sharp belief (<= .95)
        self.num_targets_tracked = []

        ###IMPORTANT####
        self.multiagent_policy = False

        # GYM and RL variables
        # Action is the displacement of the robot
        # SCENE1D in this case it is only action in X
        self.action_space_mapping = np.array([2, 2, 1]) # Scaling action factor

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
                "location": spaces.Box(np.array([-2 * np.sqrt(2*SIDE*SIDE), -2 * np.sqrt(2*SIDE*SIDE), -np.pi]), np.array([2 * np.sqrt(2*SIDE*SIDE), 2 * np.sqrt(2*SIDE*SIDE), np.pi]), shape=(3,), dtype=np.float32),
                "belief": spaces.Box(np.array([0.0]), np.array([1.0+1e-10]), shape=(1,), dtype=np.float32),
                "measurement": spaces.Box(np.array([0.0]), np.array([1.0+1e-10]), shape=(1,), dtype=np.float32),
                "velocity": spaces.Box(np.array([-np.inf,-np.inf]), np.array([np.inf,np.inf]), shape=(2,), dtype=np.float32),
                "tracked": spaces.Box(np.array([0.0]), np.array([1.0]), shape=(1,), dtype=np.float32)
            })
            self.robot_space = spaces.Dict({
                "location": spaces.Box(np.array([-2 * np.sqrt(2 * SIDE * SIDE), -2 * np.sqrt(2 * SIDE * SIDE), -np.pi]),
                                       np.array([2 * np.sqrt(2 * SIDE * SIDE), 2 * np.sqrt(2 * SIDE * SIDE), np.pi]),
                                       shape=(3,), dtype=np.float32),
               #"velocity": spaces.Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]), shape=(2,),dtype=np.float32),
            })
            # Observation is a list of targets.
            self.multiple_target_space = Repeated(self.target_space, max_len=self.MAX_TARGETS)
            # add observations of other robots
            self.multiple_robot_space = Repeated(self.robot_space, max_len=self.MAX_ROBOTS-1)
            if self.multiagent_policy:
                self.robot_observation = spaces.Tuple((self.multiple_target_space,self.multiple_robot_space))
            else:
                self.robot_observation = self.multiple_target_space
            self.observation_space.append(self.robot_observation)

        self.action_space = spaces.Tuple(self.action_space)
        self.observation_space = spaces.Tuple(self.observation_space)

        self.MAX_STEPS = config['horizon']
        self.steps = 0

        # Variables for plotting
        self.is_plotting_scene = False
        self.targets_plots = []
        self.target_colors = []
        self.robots_plots = []
        self.is_plotting_beliefs = False
        self.robot_plot_belief = 0
        self.bar_plots = []
        self.is_plotting_images = False
        self.images = []
        self.render_enabled = False

        # Episode logs
        self.episode_beliefs = []
        self.episode_visibility = []
        self.episode_entropy = []
        self.episode_poses = []
        self.success = 0
        self.nepisode = 1
        self.episode_reward = 0


        self.reward_1target = config["reward_1target"] if "reward_1target" in config else False
        #self.static_targets = True
        self.env_mode = config["env_mode"] # static / cte_vel: cte velocity / brown: brownian motion / sf_goal: social forces pedestrian dynamics
        # self.fix_seed = False
        # self.seed = 1
        self.delta_t = 0.25
        self.robot_target_collision = False
        self.robot_robot_occlusion = True

        if self.env_mode == 'cte_vel':
            self.targets_vel = []
            self.old_targets_vel = []
            self.target_maxvel = 1  # max target velocity 1m/s

        if self.env_mode == 'sf_goal':
            self.targets_t1 = []
            self.targets_vel = []
            self.targets_desvel = []
            #self.old_targets_vel = []
            #self.target_maxvel = 1  # max target velocity 1m/s
            self.targets_maxvel = []
            self.targets_goal = []
            self.socialForce_params = {
                'desvel_mean': 1.34,
                'desvel_std': 0.26,
                'potentials': {'target_cte': 2.1,#21.1, #2.1,
                               'target_sigma': 0.3,#2, #0.3
                               'wall_cte': 10,
                               'wall_R': 0.2},#0.6},  #0.2
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
        if self.visibility_hist:
            self.obs_histogram = []
            self.obs_tracked_histogram = []
            self.tracked_histogram = []

        self.save_video = False or (self.test and False)

        # robots are agents in RLlib
        self.num_agents = self.nrobots
        self.agent_ids = list(range(self.num_agents))
        self.observation_space_dict = self._make_dict(self.observation_space)
        self.action_space_dict = self._make_dict(self.action_space)

        self.observation_space = self.observation_space[0]
        self.action_space = self.action_space[0]

        #self.init_scene(nrobots=self.nrobots, ntargets=self.ntargets)

        #self.time_aux = time.time()
        self.simulated_perception = "dummy"  # dummy / simulated
        if self.simulated_perception == "simulated":
            # self.yolo_model = create_trained_yolo()
            self.unrealPedestrianData = [Pedestrian(1, real_class + 1, self.number_of_classes, load_imgs=False) for real_class in
                                         range(self.number_of_classes)]
            for pedestrian in self.unrealPedestrianData:
                pedestrian.load_probas()

    def init_scene(self, nrobots=1, ntargets=1):

        # Parameter initialization
        self.steps = 0
        self.ntargets = ntargets
        self.colors = auxFunctions.get_cmap(self.number_of_classes)
        self.targets = [np.zeros(3) for i in range(ntargets)]
        self.targets_t1 = [np.zeros(3) for i in range(ntargets)]
        self.targets_vel = [np.zeros(2) for i in range(ntargets)]
        self.targets_desvel = [0 for i in range(ntargets)]
        self.targets_maxvel = [0 for i in range(ntargets)]
        self.targets_goal = [np.zeros(2) for i in range(ntargets)]
        self.target_colors = np.random.rand(3, self.ntargets)
        self.nrobots = nrobots
        self.robots = [np.zeros(3) for r in range(nrobots)]
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
        else:
            values = 1.0 / self.number_of_classes
            self.beliefs = [values * np.ones((ntargets, self.number_of_classes)) for r in range(nrobots)]
            self.num_targets_tracked = [np.zeros(ntargets, dtype=bool) for r in range(nrobots)]

        if self.robot_target_assignment:
            self.target_assigned = [False for t in range(ntargets)]

        assert self.dimensions == 3  #otherwise bad

        if self.dimensions == 3 and (not self.load_scn or self.nepisode-2 > len(self.scenario_loaded_batch)):
            positions, _, _, _ = auxFunctions.randomPositionsProximity(N=ntargets + nrobots, side=2*(SIDE-2), dmin=MINDIST, # changed from SIDE TO SIDE-2
                                                                       dmax= np.sqrt(8*SIDE*SIDE))
            if self.env_mode == 'sf_goal':
                goals, _, _, _ = auxFunctions.randomPositionsProximity(N=ntargets, side=2 * SIDE,
                                                                       dmin=MINDIST,
                                                                       dmax=np.sqrt(8*SIDE*SIDE))
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

            self.targets_t1 = np.copy(self.targets)

            for r in range(nrobots):
                self.robots[r][0:2] = np.copy(positions[r + ntargets])
                self.robots[r][2] = np.random.uniform(0, 360)

            if self.save_scn:
                self.batch_scenario()
                if self.nepisode == self.max_episodes:
                    self.save_scenarios()
        else:
            assert self.dimensions==3
            self.load_scenario()
            self.targets_t1 = np.copy(self.targets)

    def load_scn_batch(self):
        foldername = self.load_scn_folder + '/saved_scenarios'
        foldername += '_' + str(self.nrobots)
        foldername += '_' + str(self.ntargets)
        foldername += '_' + self.env_mode
        foldername += '.csv'
        with open(foldername, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        return data

    def batch_scenario(self):
        row = [self.nepisode, self.nrobots, self.ntargets]
        row += list(np.array(self.robots).flatten())
        row += list(np.array(self.targets).flatten())
        row += list(np.array(self.targets_ID).flatten())
        if self.env_mode == 'cte_vel':
            row += list(np.array(self.targets_vel).flatten())
        elif self.env_mode == 'sf_goal':
            row += list(np.array(self.targets_desvel).flatten())
            row += list(np.array(self.targets_maxvel).flatten())
            row += list(np.array(self.targets_goal).flatten())
            row += list(np.array(self.targets_vel).flatten())

        self.scenario_saved_batch.append(row)

    def load_scenario(self):
        scn_id = self.nepisode-2
        auxbool = False
        if scn_id<0:
            scn_id=0
        else:
            auxbool = True
        current_scenario = [float(element) for element in self.scenario_loaded_batch[scn_id]]
        #print(self.nepisode-1, int(current_scenario[0]))
        if auxbool: assert self.nepisode-1 == int(current_scenario[0])
        assert self.nrobots == int(current_scenario[1])
        assert self.ntargets == int(current_scenario[2])
        auxid = 3

        self.robots = list(np.array(current_scenario[auxid:auxid + self.nrobots*3]).reshape((self.nrobots,3)))
        auxid += self.nrobots*3

        self.targets = list(np.array(current_scenario[auxid:auxid + self.ntargets * 3]).reshape((self.ntargets, 3)))
        auxid += self.ntargets * 3

        self.targets_ID = [int(element) for element in current_scenario[auxid:auxid + self.ntargets]]
        auxid += self.ntargets

        if self.env_mode == 'cte_vel':
            self.targets_vel = list(np.array(current_scenario[auxid:auxid + self.ntargets * 2]).reshape((self.ntargets, 2)))
            auxid += self.ntargets * 2

        elif self.env_mode == 'sf_goal':
            self.targets_desvel = current_scenario[auxid:auxid + self.ntargets]
            auxid += self.ntargets
            self.targets_maxvel = current_scenario[auxid:auxid + self.ntargets]
            auxid += self.ntargets
            self.targets_goal = list(np.array(current_scenario[auxid:auxid + self.ntargets * 2]).reshape((self.ntargets, 2)))
            auxid += self.ntargets * 2
            self.targets_vel = list(np.array(current_scenario[auxid:auxid + self.ntargets * 2]).reshape((self.ntargets, 2)))
            auxid += self.ntargets * 2


        #print('all well')





    def save_scenarios(self):
        foldername = self.save_scn_folder + '/saved_scenarios'
        foldername += '_'+str(self.nrobots)
        foldername += '_'+str(self.ntargets)
        foldername += '_'+self.env_mode
        foldername += '.csv'

        with open(foldername, 'a') as f:
            write = csv.writer(f)
            write.writerows(self.scenario_saved_batch)


        #trainEvlogs2csv(foldername, csvDict, csvDict['episode'])


    def init_plot_scene(self, num_figure=1):
        self.is_plotting_scene = True

        fig = plt.figure(num_figure)
        ax = fig.gca()
        ax.set_xlim((-SIDE - 1, SIDE + 1))
        ax.set_ylim((-SIDE - 1, SIDE + 1))
        ax.set_aspect('equal')

        # creating list
        for t in range(self.ntargets):
            my_item = TargetPlot(shirt_color=self.colors(self.targets_ID[t]),  # np.random.rand(3, ),
                                 shirt=mpl.patches.Wedge((self.targets[t][0], self.targets[t][1]), DIM_TARGET,
                                                         theta1=-90 + self.targets[t][2],
                                                         theta2=90 + self.targets[t][2], color='k'),
                                 face=mpl.patches.Wedge((self.targets[t][0], self.targets[t][1]), DIM_TARGET,
                                                        theta1=90 + self.targets[t][2],
                                                        theta2=270 + self.targets[t][2], color='k'))
            my_item.shirt.set_facecolor(my_item.shirt_color)
            my_item.face.fill = False
            self.targets_plots.append(my_item)
            ax.add_artist(my_item.shirt)
            ax.add_artist(my_item.face)

        for r in range(self.nrobots):
            # print('init plot robot')
            c, s = np.cos(np.radians(self.robots[r][2])), np.sin(np.radians(self.robots[r][2]))
            t = np.array([0.5, 0.4])
            R = np.array(((c, -s), (s, c)))
            shift = R.dot(t)
            # print(f'Shift = {shift}')
            my_item = RobotPlot(color=np.random.rand(3, ),
                                shape=mpl.patches.Rectangle(self.robots[r][0:2] - shift, 1, 0.8,
                                                            angle=self.robots[r][2], color='k'),
                                fov=mpl.patches.Wedge((self.robots[r][0], self.robots[r][1]), 2,
                                                      theta1=self.robots[r][2] - self.robots_fov[r],
                                                      theta2=self.robots[r][2] + self.robots_fov[r], color='k'))
            my_item.fov.set_facecolor(my_item.color)
            my_item.fov.set_alpha(0.7)
            my_item.shape.fill = False
            self.robots_plots.append(my_item)
            ax.add_artist(my_item.shape)
            ax.add_artist(my_item.fov)

        return fig, ax

    def init_plot_beliefs(self, num_figure=2, num_robot=0):
        self.is_plotting_beliefs = True

        # TODO NOT NOW current implementation only shows beliefs for robot 0 (CENTRALIZED BELIEFS)
        self.robot_plot_belief = 0
        dim_window = int(np.ceil(math.sqrt(self.ntargets)))
        tags = []
        for classes in range(self.number_of_classes):
            label = 'L' + str(classes)
            tags = np.append(tags, label)
        y_pos = np.arange(len(tags))

        fig = plt.figure(num=num_figure)
        outer = gridspec.GridSpec(dim_window, dim_window, wspace=0.1, hspace=0.1)
        ax = []
        for t in range(self.ntargets):
            ax.append(plt.Subplot(fig, outer[t]))
            ax[t].set_xlim((-0.75, self.number_of_classes - 1 + 0.75))
            ax[t].set_ylim((0, 1))
            if t >= self.ntargets - dim_window:
                ax[t].set_xticks(y_pos)  # values
                ax[t].set_xticklabels(tags)  # labels
            else:
                ax[t].set_xticks([])
            ax[t].set_yticks([])
            bar_plot = ax[t].bar(tags, self.beliefs[self.robot_plot_belief][t, :], color=self.target_colors[:, t])
            self.bar_plots.append(bar_plot)
            fig.add_subplot(ax[t])

        return fig, ax

    def init_plot_images(self, num_figure=3):
        self.is_plotting_images = True
        # beliefs will only be plotted for one robot
        dim_window = int(np.ceil(math.sqrt(self.nrobots)))
        fig, ax = plt.subplots(dim_window, dim_window, num=num_figure)
        if dim_window > 1:
            ax = ax.flatten()
        else:
            ax = [ax]

        for r in range(self.nrobots):
            ax[r].set_xlim((0, 640))
            ax[r].set_ylim((0, 480))
            image = []
            for t in range(self.ntargets):
                rect = mpatches.Polygon(np.array([[0, 0], [0, 1]]).transpose())
                image.append(rect)
                ax[r].add_artist(rect)
                ax[r].artists[t].set_color(self.colors(self.targets_ID[t]))
                ax[r].artists[t].set_visible(False)
            self.images.append(image)
        return fig, ax

    def relative_pose(self, target, robot):
        assert self.dimensions == 3
        # SCENE3D full relative pose including orientations
        c, s = np.cos(np.radians(self.robots[robot][2])), np.sin(np.radians(self.robots[robot][2]))
        robot_rot_t = np.array(((c, s), (-s, c)))
        loc = np.matmul(robot_rot_t, (self.targets[target][0:2] - self.robots[robot][0:2]).transpose())
        angle, _ = auxFunctions.angle_diff(np.radians(self.robots[robot][2]), np.radians(self.targets[target][2]))
        rel_pose = np.append(np.copy(loc), np.degrees(angle))

        return rel_pose

    def bounded_relative_pose(self, target, robot):
        assert self.dimensions == 3
        # SCENE3D full relative pose including orientations
        c, s = np.cos(np.radians(self.robots[robot][2])), np.sin(np.radians(self.robots[robot][2]))
        robot_rot_t = np.array(((c, s), (-s, c)))

        global_dist_vector = self.targets[target][0:2] - self.robots[robot][0:2]
        if np.abs(global_dist_vector[0])>OBS_BOUNDS or np.abs(global_dist_vector[1])>OBS_BOUNDS:
            if np.abs(global_dist_vector[0]) > np.abs(global_dist_vector[1]):
                global_dist_vector[1] = np.sign(global_dist_vector[0]) * OBS_BOUNDS * global_dist_vector[1] / global_dist_vector[0]
                global_dist_vector[0] = np.sign(global_dist_vector[0]) * OBS_BOUNDS
            else:
                global_dist_vector[0] = np.sign(global_dist_vector[1]) * OBS_BOUNDS * global_dist_vector[0] / global_dist_vector[1]
                global_dist_vector[1] = np.sign(global_dist_vector[1]) * OBS_BOUNDS


        loc = np.matmul(robot_rot_t, global_dist_vector.transpose())
        angle, _ = auxFunctions.angle_diff(np.radians(self.robots[robot][2]), np.radians(self.targets[target][2]))
        rel_pose = np.append(np.copy(loc), np.degrees(angle))

        return rel_pose

    def relative_pose_rXr(self, robot2, robot1):
        assert self.dimensions == 3
        # SCENE3D full relative pose including orientations
        c, s = np.cos(np.radians(self.robots[robot1][2])), np.sin(np.radians(self.robots[robot1][2]))
        robot_rot_t = np.array(((c, s), (-s, c)))
        loc = np.matmul(robot_rot_t, (self.robots[robot2][0:2] - self.robots[robot1][0:2]).transpose())
        angle, _ = auxFunctions.angle_diff(np.radians(self.robots[robot1][2]), np.radians(self.robots[robot2][2]))
        rel_pose = np.append(np.copy(loc), np.degrees(angle))

        return rel_pose

    def relative_vel(self, target, robot):
        assert self.dimensions == 3
        c, s = np.cos(np.radians(self.robots[robot][2])), np.sin(np.radians(self.robots[robot][2]))
        robot_rot_t = np.array(((c, s), (-s, c)))
        rel_vel = np.matmul(robot_rot_t, (self.targets_vel[target][0:2]).transpose())
        return rel_vel

    def rotate_action_pos(self, action_pos, rotation):
        assert self.dimensions == 3
        c, s = np.cos(np.radians(rotation)), np.sin(np.radians(rotation))
        robot_rot_t = np.array(((c, -s), (s, c)))
        rot_act = np.matmul(robot_rot_t,action_pos)
        return rot_act

    def get_projection(self, target, robot):
        # This function only makes sense in the 3D case
        if self.dimensions == 3:
            box = [[0, 0], [-DIM_TARGET, DIM_TARGET]]

            # World coordinates of the target size
            c, s = np.cos(np.radians(self.targets[target][2])), np.sin(np.radians(self.targets[target][2]))
            target_rot = np.array(((c, -s), (s, c)))
            box2 = target_rot.dot(box).transpose() + self.targets[target][0:2]
            box2 = box2.transpose()
            # print(box2)
            # fig = plt.figure(1)
            # ax = plt.gca()
            # plt.plot(box2[0][0],box2[1][0],'ro')
            # plt.plot(box2[0][1], box2[1][1], 'ro')

            # Relative position of the target size wrt to the robot frame
            c, s = np.cos(np.radians(self.robots[robot][2])), np.sin(np.radians(self.robots[robot][2]))
            robot_rot_t = np.array(((c, s), (-s, c)))
            box_relative_trans = np.matmul(robot_rot_t, (box2.transpose() - self.robots[0][0:2]).transpose())
            # print(box_relative_trans)
            # print(np.linalg.norm(box_relative_trans[:,1]-box_relative_trans[:,0]))

            # print(f'distance = {np.linalg.norm(self.targets[target][0:2] - self.robots[robot][0:2])}')

            # camera projection
            focal = 400
            p1 = Point(focal * -box_relative_trans[1][0] / box_relative_trans[0][0] + 320,
                       -focal / box_relative_trans[0][0] + 240)
            p2 = Point(focal * -box_relative_trans[1][1] / box_relative_trans[0][1] + 320,
                       -focal / box_relative_trans[0][1] + 240)
            p3 = Point(focal * -box_relative_trans[1][0] / box_relative_trans[0][0] + 320,
                       0.8 * focal / box_relative_trans[0][0] + 240)
            p4 = Point(focal * -box_relative_trans[1][1] / box_relative_trans[0][1] + 320,
                       0.8 * focal / box_relative_trans[0][1] + 240)
            try:
                # pol = Polygon([p1, p2, p4, p3])
                pol = Polygon([[p1.x, p1.y], [p2.x, p2.y], [p4.x, p4.y], [p3.x, p3.y]])
            except:
                print("something went wrong when building the polygon")

        else:
            pol = []

        return pol

    def observation(self, target, robot):
        if self.simulated_perception == "dummy":
            prob = np.ones(self.number_of_classes) / self.number_of_classes
            if self.visibility[robot, target]:
                if self.dimensions == 3:  # ASK EDUARDO
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

                if self.is_plotting_scene:  # and (robot == 0):
                    self.targets_plots[target].face.fill = False

            else:  # TARGET IS NOT VISIBLE (only update the plots)
                if (self.dimensions == 3) and self.is_plotting_images:
                    self.images[robot][target].set_visible(False)
                if self.is_plotting_scene and (robot == 0):
                    self.targets_plots[target].face.fill = True

            if self.num_targets_tracked[0][target] and self.render_enabled:
                self.targets_plots[target].shirt.set_facecolor((0.031249343749343732, 1.0, 1.3125013125390552e-06,
                                                                1.0))  # shirt_color = (0.031249343749343732, 1.0, 1.3125013125390552e-06, 1.0)
            elif len(self.targets_plots) != 0:
                self.targets_plots[target].shirt.set_facecolor(self.colors(self.targets_ID[target]))

        elif self.simulated_perception == "simulated":
            prob = np.ones(self.number_of_classes) / self.number_of_classes
            if self.visibility[robot, target]:
                rel_pose = self.relative_pose(target, robot)
                rel_pose[0:2] = np.float32([rel_pose[0:2]])
                rel_pose[2] = np.radians(np.float32([rel_pose[2]]))
                prob = self.unrealPedestrianData[self.targets_ID[target]].getProb(rel_pose)

                if rel_pose[0] > self.unrealPedestrianData[self.targets_ID[target]].max_x or \
                        rel_pose[0] < 0.5 or \
                        rel_pose[1] < self.unrealPedestrianData[self.targets_ID[target]].min_y or \
                        rel_pose[1] > self.unrealPedestrianData[self.targets_ID[target]].max_y:
                    prob = np.ones_like(prob) / self.number_of_classes

            if not np.array_equal(prob, np.ones_like(prob) / self.number_of_classes):  # aka information is visible
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
                self.targets_plots[target].shirt.set_facecolor((0.031249343749343732, 1.0, 1.3125013125390552e-06,
                                                                1.0))  # shirt_color = (0.031249343749343732, 1.0, 1.3125013125390552e-06, 1.0)
            elif len(self.targets_plots) != 0:
                self.targets_plots[target].shirt.set_facecolor(self.colors(self.targets_ID[target]))

        return prob

    def is_visible(self, target, robot):
        is_target_visible = True
        if self.dimensions == 3:
            # SCENE3D takes everything into account
            # TODO NOT NOW Missing to check if other robot blocks it DIM3

            # First check if robot is looking to the forward face
            # Relative orientation
            angle, _ = auxFunctions.angle_diff(np.radians(self.robots[robot][2]), np.radians(self.targets[target][2]))
            angle = np.degrees(angle)
            # print(angle)
            if (angle > -90) and (angle < 90):
                return False

            # Then check if it is in the fov of the robot
            bearing = math.atan2(self.targets[target][1] - self.robots[robot][1],
                                 self.targets[target][0] - self.robots[robot][0])
            angle, _ = np.degrees(
                auxFunctions.angle_diff(bearing, np.radians(self.robots[robot][2])))
            if np.abs(angle) > self.robots_fov[robot]:
                return False

            # Now check if other target blocks it
            l = LineString([self.robots[robot][0:2], self.targets[target][0:2]])
            t = 0
            while is_target_visible and (t < self.ntargets):
                if t != target:
                    p = Point(self.targets[t][0:2])
                    c = p.buffer(DIM_TARGET).boundary
                    intersections = c.intersection(l)
                    if not intersections.is_empty:
                        is_target_visible = False
                t += 1

            if self.robot_robot_occlusion:
                r = 0
                while is_target_visible and (r < self.nrobots):
                    if r != robot:
                        p = Point(self.robots[r][0:2])
                        c = p.buffer(DIM_ROBOT).boundary
                        intersections = c.intersection(l)
                        dist_tr1 = np.linalg.norm(self.robots[robot]-self.targets[target])
                        dist_tr2 = np.linalg.norm(self.robots[r]-self.targets[target])
                        dist_rr = np.linalg.norm(self.robots[r]-self.robots[robot])
                        if not intersections.is_empty:
                            if dist_rr > DIM_ROBOT:
                                is_target_visible = False
                            elif dist_tr1 > dist_tr2:
                                is_target_visible = False
                            elif dist_tr1 == dist_tr2 and robot > r:
                                is_target_visible = False
                    r += 1

        return is_target_visible

    def check_visibility(self):
        visible = np.ones((self.nrobots, self.ntargets), dtype=bool)
        # Lets check what targets the robot can see
        for r in range(self.nrobots):
            for t in range(self.ntargets):
                visible[r][t] = self.is_visible(t, r)
        return visible

    def move_one_target(self, target):

        # We generate a random small motion in the target and check feasibility
        # if the motion is feasible we update the position of the target
        # otherwise the target remains static
        # in 3D the target always rotates, even if it does not move
        # TODO NOT NOW, it'd be interesting to force the targets to maintain a minimum distance wrt to the robots (social)

        max_motion = 0.25
        max_rotation = 15
        if self.dimensions == 3:
            # SCENE3D full motion
            newpos = np.zeros(2)
            newpos[0] = self.targets[target][0] + random.uniform(-max_motion, max_motion)
            newpos[1] = self.targets[target][1] + random.uniform(-max_motion, max_motion)
            is_good_motion = (newpos[0] > -SIDE) and (newpos[0] < SIDE) \
                             and (newpos[1] > -SIDE) and (newpos[1] < SIDE)
            # check collision target-other targets
            t = 0
            while is_good_motion and (t < self.ntargets):
                if t != target:
                    distance = np.linalg.norm(self.targets[t][0:2] - newpos)
                    if distance < MINDIST:
                        is_good_motion = False
                t += 1

            # check collision target-robot
            if self.robot_target_collision:
                r = 0
                while is_good_motion and (r < self.nrobots):
                    distance = np.linalg.norm(self.robots[r][0:2] - newpos)
                    if distance < MINDIST:
                        is_good_motion = False
                    else:
                        r += 1

            if is_good_motion: # update only if the movement of the target is accepted
                self.targets[target][0] = newpos[0]
                self.targets[target][1] = newpos[1]
            self.targets[target][2] = (self.targets[target][2] + random.uniform(-max_rotation, max_rotation)) % 360

    
    def new_pos_candidates(self):
        target_newpos = []
        for target in range(self.ntargets):
            newpos = np.zeros(2)
            newpos[0] = self.targets[target][0] + self.targets_vel[target][0] * self.delta_t  # added timestep
            newpos[1] = self.targets[target][1] + self.targets_vel[target][1] * self.delta_t  # added timestep notion
            target_newpos.append(newpos)
        return target_newpos

    def move_one_target_cteVel(self, target, targetNewPos):

        # We generate a random small motion in the target and check feasibility
        # if the motion is feasible we update the position of the target
        # otherwise the target remains static
        # in 3D the target always rotates, even if it does not move
        # TODO NOT NOW, it'd be interesting to force the targets to maintain a minimum distance wrt to the robots (social)

        assert self.dimensions == 3

        # SCENE3D full motion
        newpos = targetNewPos[target]
        is_good_motion = (newpos[0] > -SIDE) and (newpos[0] < SIDE) \
                         and (newpos[1] > -SIDE) and (newpos[1] < SIDE)
        if not is_good_motion:
            if not -SIDE < newpos[0] < SIDE:
                self.targets_vel[target][0] = -self.old_targets_vel[target][0]
            if not -SIDE < newpos[1] < SIDE:
                self.targets_vel[target][1] = -self.old_targets_vel[target][1]

        # check collision target-other targets
        t = 0
        while is_good_motion and (t < self.ntargets):
            if t != target:
                distance = np.linalg.norm(targetNewPos[t][0:2] - newpos)
                if distance < MINDIST:
                    is_good_motion = False
                    #vel_sign = np.sign(self.old_targets_vel[target] * self.old_targets_vel[t])
                    #self.targets_vel[target] = self.old_targets_vel[target]*vel_sign
                    self.targets_vel[target][0] = self.old_targets_vel[t][0]
                    self.targets_vel[target][1] = self.old_targets_vel[t][1]
            t += 1

        # check collision target-robot
        if self.robot_target_collision:
            r = 0
            while is_good_motion and (r < self.nrobots):
                distance = np.linalg.norm(self.robots[r][0:2] - newpos)
                if distance < MINDIST:
                    is_good_motion = False
                else:
                    r += 1

        if is_good_motion: # update only if the movement of the target is accepted
            self.targets[target][0] = newpos[0]
            self.targets[target][1] = newpos[1]

        angle = np.angle(self.targets_vel[target][0] + self.targets_vel[target][1] * 1j, deg=True)
        if angle < 0:
            angle += 360
        self.targets[target][2] = angle

    def sf_update(self):
        """
        It computes the next candidate position according to social forces pedestrian interection
        It also updates the current velocity from all targets
        :return: vector of candidate positions for each target
        """
        target_newpos = []
        target_newvel = []
        tao = self.socialForce_params['tao']
        delta_t = self.socialForce_params['delta_t']
        V0 = self.socialForce_params['potentials']['target_cte']
        sigma = self.socialForce_params['potentials']['target_sigma']
        UB = self.socialForce_params['potentials']['wall_cte']
        R = self.socialForce_params['potentials']['wall_R']
        psi = self.socialForce_params['psi']
        cparam = self.socialForce_params['c']

        ## compute desired directions
        desired_direction = []

        delta_x = np.array([1e-10, 0])
        delta_y = np.array([0, 1e-10])
        for t in range(self.ntargets):
            desired_direction.append((self.targets_goal[t] - self.targets[t][0:2]) / np.linalg.norm(self.targets_goal[t] - self.targets[t][0:2]))

        for t1 in range(self.ntargets):
            t1pos = self.targets[t1][0:2]  # current position for t1
            t1pos_t1 = self.targets_t1[t1][0:2]  # position previous ts for t1

            force_goal_attraction = 0
            force_target_repulsion = 0
            force_wall_repulsion = 0

            ## atractive force goal
            force_goal_attraction += (self.targets_desvel[t1]*desired_direction[t1] - self.targets_vel[t1])/tao

            ## repulsive force other targets
            for t2 in range(self.ntargets):
                if t1==t2: continue
                t2pos = self.targets[t2][0:2]  # current position for t1

                centerdist = t1pos - t2pos
                dimension_dist = MINDIST * centerdist / np.linalg.norm(centerdist)
                dist_vector = t1pos - t2pos - dimension_dist

                dist_vector_deltax = dist_vector + delta_x
                dist_vector_deltay = dist_vector + delta_y

                vel_t2 = np.linalg.norm(self.targets_vel[t2]) #self.targets_desvel[t2]
                b = 0.5 * np.sqrt((np.linalg.norm(dist_vector) + np.linalg.norm(dist_vector - vel_t2*delta_t*desired_direction[t2]))**2 - (vel_t2*delta_t)**2)
                b_deltax = 0.5 * np.sqrt((np.linalg.norm(dist_vector_deltax) + np.linalg.norm(dist_vector_deltax - vel_t2 * delta_t * desired_direction[t2])) ** 2 - (vel_t2 * delta_t) ** 2)
                b_deltay = 0.5 * np.sqrt((np.linalg.norm(dist_vector_deltay) + np.linalg.norm(dist_vector_deltay - vel_t2 * delta_t * desired_direction[t2])) ** 2 - (vel_t2 * delta_t) ** 2)

                potentialV = V0*np.exp(-b/sigma)
                potentialV_deltax = V0*np.exp(-b_deltax/sigma)
                potentialV_deltay = V0 * np.exp(-b_deltay / sigma)
                dVdx = (potentialV_deltax-potentialV)/(dist_vector_deltax[0]-dist_vector[0])
                dVdy = (potentialV_deltay-potentialV)/(dist_vector_deltay[1]-dist_vector[1])
                force_repulsion = -np.array([dVdx, dVdy])
                weight = 1 if np.matmul(desired_direction[t1],-force_repulsion) >= np.linalg.norm(force_repulsion)*np.cos(np.deg2rad(psi)) else cparam

                force_target_repulsion += weight*force_repulsion

            ## repulsive force walls
            rel_pos_walls = np.zeros(4)  # relative positions to walls
            rel_pos_walls[0] = np.abs(self.targets[t1][1] - SIDE) # top
            rel_pos_walls[1] = np.abs(self.targets[t1][1] + SIDE) # bottom
            rel_pos_walls[2] = np.abs(self.targets[t1][0] - SIDE) # right
            rel_pos_walls[3] = np.abs(self.targets[t1][0] + SIDE) # left

            rel_pos_walls_delta = np.zeros(4)  # relative positions to walls + delta
            rel_pos_walls_delta[0] = np.abs(self.targets[t1][1] - SIDE - delta_y[1])  # top
            rel_pos_walls_delta[1] = np.abs(self.targets[t1][1] + SIDE + delta_y[1])  # bottom
            rel_pos_walls_delta[2] = np.abs(self.targets[t1][0] - SIDE - delta_x[0])  # right
            rel_pos_walls_delta[3] = np.abs(self.targets[t1][0] + SIDE + delta_x[0])  # left

            # compute potentials
            for i in range(4):
                potentialU = UB*np.exp(-rel_pos_walls[i]/R)
                potentialU_delta = UB*np.exp(-rel_pos_walls_delta[i]/R)

                dUdx = (potentialU_delta-potentialU)/(rel_pos_walls_delta[i]-rel_pos_walls[i]) if (i==2 or i==3) else 0
                dUdy = (potentialU_delta-potentialU)/(rel_pos_walls_delta[i]-rel_pos_walls[i]) if (i==0 or i==1) else 0

                force_repulsion = -np.array([dUdx,dUdy])
                if i==0:
                    force_repulsion[1] = -force_repulsion[1]
                if i==2:
                    force_repulsion[0] = -force_repulsion[0]

                weight = 1 if np.matmul(desired_direction[t1], -force_repulsion) >= np.linalg.norm(
                    force_repulsion) * np.cos(np.deg2rad(psi)) else cparam
                force_wall_repulsion += weight*force_repulsion

            delta_w = (force_goal_attraction + force_target_repulsion + force_wall_repulsion)*self.delta_t
            candidate_vel = self.targets_vel[t1] + delta_w
            g = 1 if np.linalg.norm(candidate_vel) <= self.targets_maxvel[t1] else self.targets_maxvel[t1] / np.linalg.norm(candidate_vel)

            target_newvel.append(g*candidate_vel)
            target_newpos.append(self.targets[t1][0:2] + target_newvel[t1]*self.delta_t)

        self.targets_vel = target_newvel
        return target_newpos

    def move_one_target_sforces(self, target, targetNewPos):
        """
         We generate a random small motion in the target and check feasibility
        if the motion is feasible we update the position of the target
        otherwise the target remains static
        in 3D the target always rotates, even if it does not move
        TODO NOT NOW, it'd be interesting to force the targets to maintain a minimum distance wrt to the robots (social)
        """
        newpos = targetNewPos[target]
        assert self.dimensions == 3

        # SCENE3D full motion --> Check that the new position does not collide

        is_good_motion = (newpos[0] > -SIDE) and (newpos[0] < SIDE) \
                         and (newpos[1] > -SIDE) and (newpos[1] < SIDE)
        # check collision target-other targets
        t = 0
        while is_good_motion and (t < self.ntargets):
            if t != target:
                distance = np.linalg.norm(self.targets[t][0:2] - newpos)
                if distance < MINDIST:
                    is_good_motion = False
            t += 1

        # check collision target-robot
        if self.robot_target_collision:
            r = 0
            while is_good_motion and (r < self.nrobots):
                distance = np.linalg.norm(self.robots[r][0:2] - newpos)
                if distance < MINDIST:
                    is_good_motion = False
                else:
                    r += 1

        if is_good_motion: # update only if the movement of the target is accepted
            self.targets[target][0] = newpos[0]
            self.targets[target][1] = newpos[1]

        angle = np.angle(self.targets_vel[target][0] + self.targets_vel[target][1] * 1j, deg=True)
        if angle < 0:
            angle += 360
        self.targets[target][2] = angle


    def new_target_goals(self, t):

        """
        For social forces dynamics: Decide if a target has arrived to its goal, then decide a new one
        """
        goal_dist = np.linalg.norm(self.targets[t][0:2]-self.targets_goal[t])
        if goal_dist < MINDIST:
            valid_new_goal = False  # needs to be initialized
            new_goal = np.random.uniform(-8,8,2)  # needs to be initialized

            while not valid_new_goal:
                new_goal = np.random.uniform(-8, 8, 2)
                valid_new_goal = True
                for t2 in range(self.ntargets):
                    if t==t2: continue
                    if np.linalg.norm(self.targets_goal[t2]-new_goal) <= MINDIST:
                        valid_new_goal = False
                        break
            self.targets_goal[t] = new_goal

    def move_targets(self):
        """
        Move and update targets positions according to the chosen dynamics. First candidate positions are chosen,
        then update the target's positions if there are no collisions.
        """
        if self.env_mode =='cte_vel':
            newPosCand = self.new_pos_candidates()

        elif self.env_mode == 'sf_goal':
            newPosCand = self.sf_update()
            self.targets_t1 = np.copy(self.targets)

        for t in range(self.ntargets):
            if self.env_mode != 'static':
                if self.env_mode == 'brown':
                    self.move_one_target(t)
                if self.env_mode == 'cte_vel':
                    self.move_one_target_cteVel(t, newPosCand)
                if self.env_mode == 'sf_goal': # TODO NOT NOW testing
                    self.move_one_target_sforces(t, newPosCand)

                self.new_target_goals(t)  # update target goals in case they have been reached

            if self.is_plotting_scene:
                ce = (self.targets[t][0], self.targets[t][1])
                self.targets_plots[t].shirt.set_center(ce)
                self.targets_plots[t].shirt.set_theta1(-90 + self.targets[t][2])
                self.targets_plots[t].shirt.set_theta2(90 + self.targets[t][2])
                self.targets_plots[t].face.set_center(ce)
                self.targets_plots[t].face.set_theta1(90 + self.targets[t][2])
                self.targets_plots[t].face.set_theta2(270 + self.targets[t][2])

    def update_beliefs(self):
        """
        Update the beliefs using the observations from robot r, then update rendered plots (if render option is enabled)
        """
        for r in range(self.nrobots):
            for t in range(self.ntargets):
                prob = self.observation(t, r)
                bel = self.beliefs[r][t]

                new_belief = prob * bel / (prob * bel + (1 - prob) * (1 - bel))
                self.beliefs[r][t] = new_belief

        if self.is_plotting_beliefs:
            for t in range(self.ntargets):
                self.bar_plots[t][0].set_height(1 - self.beliefs[self.robot_plot_belief][t])
                self.bar_plots[t][1].set_height(self.beliefs[self.robot_plot_belief][t])

    def is_valid_position(self, robot, location):
        """
        Check if location is valid
        inside the environment
        at minimum distance from all the other entities
        TODO NOT NOW, for the moment only cares about targets, robots are missing, CONSIDER ROBOT-TO-ROBOT COLLISION
        """

        is_valid = (location[0] > -SIDE) and (location[0] < SIDE) \
                   and (location[1] > -SIDE) and (location[1] < SIDE)

        if self.robot_target_collision:
            t = 0
            while is_valid and (t <= self.ntargets - 1):
                distance = np.linalg.norm(self.targets[t][0:2] - location[0:2])
                if distance <= MINDIST:
                    return False
                else:
                    t += 1
        return is_valid

    def place_target(self, target, location):
        """
        Update target's positon and update it's position in the plots
        """
        self.targets[target] = np.copy(location)
        t = target
        ce = (self.targets[t][0], self.targets[t][1])
        self.targets_plots[t].shirt.set_center(ce)
        self.targets_plots[t].shirt.set_theta1(-90 + self.targets[t][2])
        self.targets_plots[t].shirt.set_theta2(90 + self.targets[t][2])
        self.targets_plots[t].face.set_center(ce)
        self.targets_plots[t].face.set_theta1(90 + self.targets[t][2])
        self.targets_plots[t].face.set_theta2(270 + self.targets[t][2])

    def place_robot(self, robot, location):
        """
        For robot "robot", we check if the position is valid and move the robot
        otherwise we simply apply the rotation command. Then update the position in the plots
        """
        if self.is_valid_position(robot, location):
            self.robots[robot] = np.copy(location)
        else:
            self.robots[robot][2] = location[2] % 360

        if self.is_plotting_scene:
            c, s = np.cos(np.radians(self.robots[robot][2])), np.sin(np.radians(self.robots[robot][2]))
            t = np.array([0.5, 0.4])
            R = np.array(((c, -s), (s, c)))
            shift = R.dot(t)

            ce = (self.robots[robot][0], self.robots[robot][1])
            self.robots_plots[robot].shape.set_xy(ce - shift)
            self.robots_plots[robot].shape.angle = self.robots[robot][2]
            self.robots_plots[robot].fov.set_center(ce)
            self.robots_plots[robot].fov.set_theta1(self.robots[robot][2] - self.robots_fov[robot])
            self.robots_plots[robot].fov.set_theta2(self.robots[robot][2] + self.robots_fov[robot])

    # FUNCTION FOR GYM AND RL
    def reset(self):
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
        if len(self.nmaxtargets)==1:
            self.ntargets = self.nmaxtargets[-1]
        else:
            self.ntargets = np.random.randint(self.nmaxtargets[0], self.nmaxtargets[1]+1) if self.test == False else self.nmaxtargets  # np.random.randint(1,self.MAX_TARGETS+1)

        if len(self.nmaxrobots)==1:
            self.nrobots = self.nmaxrobots[-1]
        else:
            self.nrobots = np.random.randint(self.nmaxrobots[0], self.nmaxrobots[1]+1) if self.test == False else self.nmaxrobots  # np.random.randint(1,self.MAX_TARGETS+1)

        # Initialize episode number and success log
        self.nepisode += 1
        self.success = 0

        if self.heuristic_target_order:
            self.targets_id_order = [[] for r in range(self.nrobots)]

        # If previously specified, use saved images to generate a .gif of the episode
        if self.save_video and self.nepisode>2:
            camfolder = self.video_foldername+'camera/'
            belfolder = self.video_foldername+'beliefs/'
            statefolder = self.video_foldername + 'state/'
            videosource = ['camera','beliefs','state']
            def make_gif(frame_folder,outfolder,videosource):
                import glob
                from PIL import Image
                imagelist = glob.glob(f"{frame_folder}/*.png")
                sortedimagelist = sorted(imagelist, key=lambda x: int(x.split('/')[-1][:-4]))
                frames = [Image.open(image) for image in sortedimagelist]
                frame_one = frames[0]
                frame_one.save(outfolder+videosource+".gif", format="GIF", append_images=frames,
                               save_all=True, duration=125, loop=0)#duration=250, loop=0)

            for i,video_folder in enumerate([camfolder,belfolder,statefolder]):
                make_gif(video_folder,self.video_foldername,videosource[i])

        # Reset the state of the environment to an initial state with the specified nrobots and ntargets
        # TODO: DONE update this function to randomize the number of robots as well.
        self.init_scene(nrobots=self.nrobots, ntargets=self.ntargets)

        # Track target visibility (mostly used to decide whether a new observation is obtained from a target or not)
        self.visibility = self.check_visibility()

        # Generate new probability estimates over each target over the environment. TODO: DONE update to multirobot
        for r in range(self.nrobots):
            for t in range(self.ntargets):
                self._observation[t, r] = self.observation(t, r)

        # Generate a new observation over the environment. TODO: DONE update to multirobot
        obs = self._obs()

        #return obs
        obs_dict = self._make_dict(obs)

        gc.collect() ### make sure it works
        return obs_dict

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
        for r in range(self.nrobots): # compute observation for each robot
            # target observations
            target_obs = []
            for t in range(self.ntargets):  # range(self.nrobots):
                aux = self.target_space.sample()
                loc = self.relative_pose(t, r)
                #loc = self.bounded_relative_pose(t, r) ## TRYING FOR SCALABILITY DURING TESTING
                aux['location'][0:2] = np.float32([loc[0:2]])

                # aux['location'][2] = np.radians(np.float32([loc[2]]))
                aux['location'][2] = np.radians(np.float32([loc[2]]))
                rel_vel = self.relative_vel(t,r)
                aux['velocity'] = np.float32([rel_vel])[0]
                aux["belief"] = np.float32([1.0-distrib.entropy(self.beliefs[0][t, :], base=2)/distrib.entropy([1/self.number_of_classes]*self.number_of_classes, base=2)+1e-10])
                distrib.entropy(self.beliefs[0][t, :], base=2)
                aux["measurement"] = np.float32([1-distrib.entropy(self._observation[t, r], base=2)/distrib.entropy([1/self.number_of_classes]*self.number_of_classes, base=2)+1e-10])
                aux["tracked"] = np.float32([np.any(self.beliefs[0][t,:] > 0.95)])
                # print(obs)
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

            # observations on the locations of other robots
            if self.multiagent_policy:
                robot_obs = []
                for r_other in range(self.nrobots):
                    if r_other == r: continue
                    aux_robot = self.robot_space.sample()
                    loc_robot = self.relative_pose_rXr(r_other, r)
                    aux_robot['location'][0:2] = np.float32([loc_robot[0:2]])
                    aux_robot['location'][2] = np.radians(np.float32([loc_robot[2]]))
                    robot_obs.append(aux_robot)
                obs.append([target_obs,robot_obs])
            else:
                obs.append(target_obs)

        # For logging purposes: simultaneous observations (total and tracked) + total targets tracked
        if self.visibility_hist:
            self.obs_histogram.append(obsed_targets)
            self.obs_tracked_histogram.append(obsed_targets_tracked)
            self.tracked_histogram.append(tracked)

        return obs


    #def step(self, action, envtest=False): # TODO NOT NOW: turn logs off if not specified (less RAM consumed)
    def step(self, action_dict, envtest=False): # RLlib adaptation
        #print(time.time()-self.time_aux)
        """
        Takes the action from the robot and runs a step of the environment.
        It returns the next observation, the current reward and the terminal signal.
        """
        ## RLlib adaptation
        action = list(action_dict.values())

        # Update timestep indicator
        # TODO DONE step does everything (moving and inference) only for the first robot --> adapt to multiple
        self.steps += 1

        # Save previous target velocity, used for target dynamics computation
        self.old_targets_vel = np.copy(self.targets_vel)

        # Move the targets
        if not envtest:
            self.move_targets()  # move targets according to dynamic model "self.env_mode"

        # We apply the action to robot 0, moving it to its current chosen location in the neighborhood if possible
        # TODO: DONE adapt to multiple robots
        for r in range(self.nrobots):
            # map actions to actual action space.
            # Map the action space to the circle of unit radius TODO: DONE adapt to obtain a list/dict of actions associated to each robot.
            # map actions to actual action space.
            # if np.linalg.norm(action[r][0:2]) > 1: action[r][0:2] = action[r][0:2] / np.linalg.norm(action[r][0:2]) # Does not work if only this is considered
            action[r] = action[r] * self.action_space_mapping

            location = np.copy(self.robots[r])
            if self.env_mode == 'cte_vel' or self.env_mode=='sf_goal' or self.env_mode=='brown' or self.env_mode=='static':
                #applied_action = np.clip(action, self.action_space.low*self.delta_t, self.action_space.high*self.delta_t)
                applied_action=action[r]*self.delta_t
            else:
                applied_action = action[r]
            location[0:2] += self.rotate_action_pos(applied_action[0:2],location[2]) #action[0:2]
            location[2] = (location[2] + 60 * applied_action[2]) % 360
            self.place_robot(r, location)

        # Now all positions from robots and targets are updated. Next: obtain new observation
        # Initialize entropy measurements (for observation and rewards) and logging variables
        num_targets_tracked = 0
        entropy_measurements = 0
        entropy_beliefs = 0
        new_targets_tracked = 0

        # Check visibility of targets, reused when obtaining probability estimates on target classes
        self.visibility = self.check_visibility()

        # We obtain probability estimates and update beliefs on target class for all the targets
        #TODO: DONE this is only for one robot -> update for multiple robots
        for t in range(self.ntargets):
            new_belief = self.beliefs[0][t, :].copy()
            for r in range(self.nrobots):
                # We take probability estimates of the target t from the new location
                self._observation[t, r] = self.observation(t, r)

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

            # Keep track of tracked targets by robot 0. TODO: DONE If centralized, this is not needed to be updated
            if not self.reward_1target or t==0:
                if np.max(self.beliefs[0][t, :]) >= 0.95:
                #if self.beliefs[0][t, self.targets_ID[t]] >= 0.95:  # MODIFIED!!! TODO NEED TO TRY WITH POSSIBILITY TO BE WRONG!!
                    if not self.num_targets_tracked[0][t]:
                        self.num_targets_tracked[0][t] = True
                        new_targets_tracked += 1
                    num_targets_tracked += 1

        #print("visibility:", self.visibility)
        #print("new observation", self._observation)
        # Reward computation. TODO: NOT NOW this could go in its own function
        # Initialization of the current reward
        reward = 0

        for r in range(self.nrobots):
            # Movement reward (punishment)
            reward += -0.01 * np.linalg.norm(action[r][0:2])
            reward += -0.01*np.linalg.norm(action[r][2])

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
            reward += -0.3
            # Variable penalty based on the entropy of the measurement (dense reward)
            if not math.isnan(entropy_beliefs) and len(self.episode_entropy) > 0:
                reward += self.episode_entropy[-1] - entropy_beliefs
                #print("entropy reward:", self.episode_entropy[-1] - entropy_beliefs)

        # Compute the observations given to the policy. TODO: (DONE) it has to return a list/dict and be adapted to multiple robots.
        obs = self._obs()

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

        #self.time_aux = time.time()
        return obs_dict, rew_dict, done_dict, info_dict

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
            foldername = foldername + '_' + str(self.MAX_STEPS)
        if self.visibility_hist:
            foldername = foldername + '_histogram'
        if self.robot_robot_occlusion:
            foldername = foldername + '_robrobOcclusion'
        if self.robot_target_assignment:
            foldername = foldername + '_rotarAssign'
        if SIDE != 8.0:
            foldername = foldername + '_' + str(SIDE)


        foldername = foldername + '_'+self.env_mode
        foldername = foldername + '.csv'
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
        print(f'Belief: {self.beliefs}')

        # Creates the folder where the images and videos are respectively stored and generated.
        self.video_foldername = self.log_folder+'/videos/'+str(self.ntargets)+'_targets_episode_' + str(self.nepisode)+'/'
        if not os.path.isdir(self.video_foldername) and self.save_video:
            os.makedirs(self.video_foldername+'camera/')
            os.makedirs(self.video_foldername + 'state/')
            os.makedirs(self.video_foldername + 'beliefs/')

        # Only render if we are testing the policy
        if self.test:
            # Initialize the window used for rendering
            if not self.render_enabled:
                self.plot_scene = True
                self.plot_beliefs = True
                self.plot_image = False
                if self.plot_scene:
                    self.fig, self.ax = self.init_plot_scene(num_figure=1)
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

            if self.save_video and self.steps>1:
                if self.plot_image: self.fig3.savefig(self.video_foldername+'camera/'+str(self.steps)+'.png')
                if self.plot_scene: self.fig.savefig(self.video_foldername + 'state/' + str(self.steps) + '.png')
                if self.plot_beliefs: self.fig2.savefig(self.video_foldername + 'beliefs/' + str(self.steps) + '.png')


    # This function can be improved but it serves its purpose
    def plot_episode(self, num_figure=1, file_name=None, show_all_poses=False):
        """
        CURRENTLY NOT USED.
        """
        fig = plt.figure(num=num_figure)
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        fig.suptitle('Semantic Information Gathering of Multiple Targets', fontsize=20)
        outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
        ax0 = plt.Subplot(fig, outer[0])
        ax0.set_title('Environment')
        ax0.set_xlim((-SIDE - 1, SIDE + 1))
        ax0.set_ylim((-SIDE - 1, SIDE + 1))
        ax0.set_aspect('equal')
        fig.add_subplot(ax0)
        ax1 = plt.Subplot(fig, outer[1])
        ax1.set_title('Target beliefs')
        ax1.axis('off')
        fig.add_subplot(ax1)

        dim_window = int(np.ceil(math.sqrt(self.ntargets)))
        inner = gridspec.GridSpecFromSubplotSpec(dim_window, dim_window, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
        axes = []
        tags = []
        for labels in range(self.number_of_classes):
            label = 'L' + str(labels)
            tags = np.append(tags, label)
        # tags = ['0', '1']
        y_pos = np.arange(len(tags))
        self.robot_plot_belief = 0
        for t in range(self.ntargets):
            axes.append(plt.Subplot(fig, inner[t]))
            axes[t] = plt.Subplot(fig, inner[t])
            axes[t].set_xlim((-0.75, self.number_of_classes - 1 + 0.75))
            axes[t].set_ylim((0, 1))
            if t >= self.ntargets - dim_window:
                axes[t].set_xticks(y_pos)  # values
                axes[t].set_xticklabels(tags)  # labels
            else:
                axes[t].set_xticks([])
            axes[t].set_yticks([])
            axes[t].bar(tags, self.beliefs[self.robot_plot_belief][t, :], color=self.target_colors[:, t])
            fig.add_subplot(axes[t])

        trajectories = np.array(self.episode_poses)
        color_targets = np.random.rand(3, self.ntargets)
        for t in range(self.ntargets):
            # color = np.random.rand(3, )
            traje = trajectories[:, t, 0:2]
            ax0.plot(traje[:, 0], traje[:, 1], color=self.target_colors[:, t], alpha=0.9, linewidth=4)
            if show_all_poses:
                list_poses = range(len(self.episode_poses))
            else:
                list_poses = range(len(self.episode_poses)-1,len(self.episode_poses))
            for it in list_poses:
                pose_target = trajectories[it, t, :]
                my_item = TargetPlot(shirt_color=self.colors(self.targets_ID[t]),
                                     shirt=mpl.patches.Wedge((pose_target[0], pose_target[1]), DIM_TARGET,
                                                             theta1=-90 + pose_target[2],
                                                             theta2=90 + pose_target[2], color='k'),
                                     face=mpl.patches.Wedge((pose_target[0], pose_target[1]), DIM_TARGET,
                                                            theta1=90 + pose_target[2],
                                                            theta2=270 + pose_target[2], color='k'))
                my_item.shirt.set_facecolor(my_item.shirt_color)
                my_item.face.set_facecolor((1.0, 1.0, 1.0))
                # my_item.face.fill = False
                my_item.shirt.set_alpha(0.2)
                my_item.face.set_alpha(0.2)
                self.targets_plots.append(my_item)
                ax0.add_artist(my_item.shirt)
                ax0.add_artist(my_item.face)
            my_item.shirt.set_alpha(1)
            my_item.face.set_alpha(1)

        trajectories = np.array(self.episode_poses)
        for r in range(self.nrobots):
            color = np.random.rand(3, )
            traje = trajectories[:, self.ntargets + r, 0:2]
            ax0.plot(traje[:, 0], traje[:, 1], '-o', color=color)
            if show_all_poses:
                list_poses = range(len(self.episode_poses))
            else:
                list_poses = range(len(self.episode_poses) - 1, len(self.episode_poses))
            for it in list_poses:
                pose_robot = trajectories[it, self.ntargets + r, :]
                c, s = np.cos(np.radians(pose_robot[2])), np.sin(np.radians(pose_robot[2]))
                t = np.array([0.5, 0.4])
                R = np.array(((c, -s), (s, c)))
                shift = R.dot(t)
                my_item = RobotPlot(color=color,
                                    shape=mpl.patches.Rectangle(pose_robot[0:2] - shift, 1, 0.8,
                                                                angle=pose_robot[2], color='k'),
                                    fov=mpl.patches.Wedge((pose_robot[0], pose_robot[1]), 2,
                                                          theta1=pose_robot[2] - self.robots_fov[r],
                                                          theta2=pose_robot[2] + self.robots_fov[r], color='k'))
                my_item.fov.set_facecolor(my_item.color)
                my_item.fov.set_alpha(0.2)
                my_item.shape.set_alpha(0.2)
                my_item.shape.fill = False
                self.robots_plots.append(my_item)
                ax0.add_artist(my_item.shape)
                ax0.add_artist(my_item.fov)
            my_item.shape.set_alpha(1)
            my_item.fov.set_alpha(0.7)

        if file_name is not None:
            fig.set_size_inches(8, 6)
            fig.canvas.draw()
            fig.canvas.flush_events()
            path_save = 'results/figure_' + str(num_figure) + '.png'
            path_save = 'results/' + file_name + '.png'
            plt.savefig(path_save, dpi=fig.dpi)
        else:
            fig.show()


    def plot_video_episode(self, file_name='video', show_all_poses=False):
        """
        CURRENTLY NOT USED
        """
        # plt.ion()
        plt.ioff()
        fig = plt.figure(figsize=(8, 6))
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        fig.suptitle('Semantic Information Gathering of Multiple Targets', fontsize=20)
        outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
        ax0 = plt.Subplot(fig, outer[0])
        ax0.set_title('Environment')
        ax0.set_xlim((-SIDE - 1, SIDE + 1))
        ax0.set_ylim((-SIDE - 1, SIDE + 1))
        ax0.set_aspect('equal')
        fig.add_subplot(ax0)
        ax1 = plt.Subplot(fig, outer[1])
        ax1.set_title('Target beliefs')
        ax1.axis('off')
        fig.add_subplot(ax1)

        dim_window = int(np.ceil(math.sqrt(self.ntargets)))
        inner = gridspec.GridSpecFromSubplotSpec(dim_window, dim_window, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
        axes = []
        tags = []
        for classes in range(self.number_of_classes):
            label = 'L' + str(classes)
            tags = np.append(tags, label)
        # tags = ['0', '1']
        y_pos = np.arange(len(tags))
        for t in range(self.ntargets):
            axes.append(plt.Subplot(fig, inner[t]))
            axes[t] = plt.Subplot(fig, inner[t])
            axes[t].set_xlim((-0.75, self.number_of_classes - 1 + 0.75))
            axes[t].set_ylim((0, 1))
            if t >= self.ntargets - dim_window:
                axes[t].set_xticks(y_pos)  # values
                axes[t].set_xticklabels(tags)  # labels
            else:
                axes[t].set_xticks([])
            axes[t].set_yticks([])
            fig.add_subplot(axes[t])

        camera = Camera(fig)

        color_targets = np.random.rand(3, self.ntargets)
        color_robots = np.random.rand(3, self.nrobots)
        trajectories = np.array(self.episode_poses)
        for it2 in range(MAX_STEPS):  # self.steps):
            if it2 >= self.steps:
                it = self.steps - 1
            else:
                it = it2
            beliefs = self.episode_beliefs[it][0]
            visibility = self.episode_visibility[it][0]
            for t in range(self.ntargets):
                pose_target = trajectories[it, t, :]
                my_item = TargetPlot(shirt_color=self.colors(self.targets_ID[t]),
                                     shirt=mpl.patches.Wedge((pose_target[0], pose_target[1]), DIM_TARGET,
                                                             theta1=-90 + pose_target[2],
                                                             theta2=90 + pose_target[2], color='k'),
                                     face=mpl.patches.Wedge((pose_target[0], pose_target[1]), DIM_TARGET,
                                                            theta1=90 + pose_target[2],
                                                            theta2=270 + pose_target[2], color='k'))
                my_item.shirt.set_facecolor(my_item.shirt_color)
                if visibility[t]:
                    my_item.face.fill = False
                else:
                    my_item.face.fill = True
                if self.dimensions == 1:
                    ax0.axvline(pose_target[0], color=color_targets[:, t])
                if show_all_poses:
                    trajectories = np.array(self.episode_poses)
                    traje = trajectories[0:it, t, 0:2]
                    ax0.plot(traje[:, 0], traje[:, 1], color=self.target_colors[:, t], alpha=0.9, linewidth=4)
                ax0.add_patch(my_item.shirt)
                ax0.add_patch(my_item.face)
                probs = beliefs[t]  # self.beliefs[0][t, :] #1 - beliefs[t], beliefs[t]]
                axes[t].bar(tags, probs, color=color_targets[:, t])
            for r in range(self.nrobots):
                color = color_robots[:, r]
                pose_robot = trajectories[it, self.ntargets + r, :]
                c, s = np.cos(np.radians(pose_robot[2])), np.sin(np.radians(pose_robot[2]))
                t = np.array([0.5, 0.4])
                R = np.array(((c, -s), (s, c)))
                shift = R.dot(t)
                my_item = RobotPlot(color=color,
                                    shape=mpl.patches.Rectangle(pose_robot[0:2] - shift, 1, 0.8,
                                                                angle=pose_robot[2], color='k'),
                                    fov=mpl.patches.Wedge((pose_robot[0], pose_robot[1]), 2,
                                                          theta1=pose_robot[2] - self.robots_fov[r],
                                                          theta2=pose_robot[2] + self.robots_fov[r], color='k'))
                my_item.fov.set_facecolor(my_item.color)
                my_item.shape.fill = False
                my_item.shape.set_alpha(1)
                my_item.fov.set_alpha(0.7)
                if self.dimensions == 1:
                    ax0.axvline(pose_robot[0], color=color_robots[:, r])
                if show_all_poses:
                    trajectories = np.array(self.episode_poses)
                    traje = trajectories[0:it, self.ntargets+r, 0:2]
                    ax0.plot(traje[:, 0], traje[:, 1], '-o', color=color_robots[:, r], alpha=0.9, linewidth=4)
                ax0.add_artist(my_item.shape)
                ax0.add_artist(my_item.fov)
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            camera.snap()
            # time.sleep(0.1)

        # TODO: NOT NOW if we want to show but not save, then it does not work...
        camera_anim = camera.animate(interval=330)
        if file_name is not None:
            path_save = 'results/' + file_name + '.mp4'
            camera_anim.save(path_save)
        else:
            plt.show()

    def _make_dict(self,values):
        return dict(zip(self.agent_ids, values))

    def seed(self,seed):
        print(seed)




if __name__ == "__main__":
    """
    Used for testing the environment
    """
    config = {
        "dimensions": 3,
        "ntargets": [1],
        "nrobots": [1],
        "nclasses": 2,
        "MAXTARGETS":10,
        "horizon": 1000,
        "env_mode": "cte_vel",
        "test": False,
        "heuristic_target_order": False,
        "reward_1target":False,
        "random_beliefs": False,
        "random_static_dynamic":True,
        "reverse_heuristic_target_order": False
    }

    # Initialize plots stuff
    plt.close('all')
    plt.ion()

    print('Init Scene')
    sc = SceneEnv(config)
    sc.reset()
    sc.robots[0] = np.array([8.0, 0.0, 180.0])
    #sc.robots[1] = np.array([0, 0.0, 180.0])
    sc.targets[0] = np.array([-8.0, 0.0, 0.0])
    # sc.targets_ID[0] = 0
    print(sc.targets_ID[0])

    plot_scene = True
    if plot_scene:
        fig, ax = sc.init_plot_scene(num_figure=1)
        # camera = Camera(fig)

    plot_beliefs = True
    if plot_beliefs:
        fig2, ax2 = sc.init_plot_beliefs(num_figure=2)
    plot_image = True
    if plot_image:
        fig3, ax3 = sc.init_plot_images(num_figure=3)

    if plot_image or plot_beliefs or plot_scene:
        plt.show()

    # MAIN LOOP
    final = False
    taux = 0
    sc.render_enabled = True
    # we ran one full episode with random actions to check that everything works
    while taux <= 100000:
        action = {'0':np.array([0.25, 0, 0]),'1':np.array([0, 0, 0.25])}#np.random.uniform(-1, 1, 3)
        #action[0] = 0
        #action[2] = 0
        obs, reward, final, _ = sc.step(action) #, envtest = True)
        print(obs)
        print(sc.beliefs[0])
        print(sc._observation)
        print(reward)

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

        if plot_image or plot_beliefs or plot_scene:
            time.sleep(0.1)

        if (taux % 400 == 0 and taux!=0) or all(final):
            sc.reset()
        taux +=1

    # sc.plot_episode(num_figure=1, file_name='caca')

    # name = 'aux'
    # im = sc.plot_video_episode(file_name=name, show_all_poses=False)



