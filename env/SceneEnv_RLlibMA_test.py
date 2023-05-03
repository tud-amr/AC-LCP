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


## SIMULATED DRONE



# Constants (TODO NOT NOW to edit them from the main file)
MAXTARGETS = 8
MAX_STEPS = 100
SIDE = 25 #25.0 #25 #8.0
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
class SceneEnv(MAScene_Base):
    def __init__(self, config):
        MAScene_Base.__init__(self, config)

        self.robot_target_assignment = True and self.heuristic_target_order

        ###IMPORTANT####
        self.multiagent_policy = False

        # GYM and RL variables
        # Action is the displacement of the robot
        # SCENE1D in this case it is only action in X
        self.action_space_mapping = np.array([2, 2, 1]) # Scaling action factor

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

            self.total_target_space = spaces.Dict({"unidentified": self.multiple_target_space, "identified": self.multiple_target_space,
                                                      "all": self.multiple_target_space})
            # add observations of other robots
            self.multiple_robot_space = Repeated(self.robot_space, max_len=self.MAX_ROBOTS - 1)

            # target observations deglossed into identified / unidentified and all.
            if self.deglossed_obs:
                if self.multiagent_policy:
                    self.robot_observation = spaces.Dict({"targets": self.total_target_space, "robots": self.multiple_robot_space})
                else:
                    self.robot_observation = spaces.Dict({"targets": self.total_target_space})
                self.observation_space.append(self.robot_observation)
            else:
                if self.multiagent_policy:
                    self.robot_observation = spaces.Tuple((self.multiple_target_space, self.multiple_robot_space))
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

        self.save_video = False or (self.test and False)

        # robots are agents in RLlib
        self.num_agents = self.nrobots
        self.agent_ids = list(range(self.num_agents))
        self.observation_space_dict = self._make_dict(self.observation_space)
        self.action_space_dict = self._make_dict(self.action_space)

        self.observation_space = self.observation_space[0]
        self.action_space = self.action_space[0]

        # enable/disable realistic drone dynamics
        self.realistic_Dengine = False
        if self.realistic_Dengine:
            from realistic_drone.py_g2g_drone_sim.test_gym.BasicDroneDynamics import DroneEnv as droneEngine
            self.drone_engines = [droneEngine(ndrones = 1, ndynamic_obstacles =0) for i in range(self.nrobots)]

        self.dyn_sigma=0.0

        #self.init_scene(nrobots=self.nrobots, ntargets=self.ntargets)

        #self.time_aux = time.time()
        self.simulated_perception = "dummy" # dummy / simulated
        if self.simulated_perception == "simulated":
            #self.yolo_model = create_trained_yolo()
            self.unrealPedestrianData = [Pedestrian(1,real_class+1,self.number_of_classes) for real_class in range(self.number_of_classes)]
            for pedestrian in self.unrealPedestrianData:
                pedestrian.load_probas()

    def init_scene(self, nrobots=1, ntargets=1):

        # Parameter initialization
        self.steps = 0
        self.hl_steps = 0
        self.last_action = None
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
        self.episode_results = None
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
                if self.realistic_Dengine:
                    self.drone_engines[r].resetFromPos(np.array([self.robots[r][0:2]]))

            if self.save_scn:
                self.batch_scenario()
                if self.nepisode == self.max_episodes:
                    self.save_scenarios()
        else:
            assert self.dimensions==3
            self.load_scenario(random_targetID=True)
            self.targets_t1 = np.copy(self.targets)

    # FUNCTION FOR GYM AND RL

    #def step(self, action, envtest=False): # TODO NOT NOW: turn logs off if not specified (less RAM consumed)
    def step(self, action_dict, envtest=False): # RLlib adaptation
        #print(time.time()-self.time_aux)
        """
        Takes the action from the robot and runs a step of the environment.
        It returns the next observation, the current reward and the terminal signal.
        """
        #print("step",self.steps)
        ## RLlib adaptation
        high_level_policy_update = self.steps % self.lh_ratio == 0
        if high_level_policy_update:
            action = list(action_dict.values())
            self.last_action = action.copy()
            # print("HIGH LEVEL POLICY!!!",action)
        else:
            action = self.last_action.copy()

        # Update timestep indicator
        self.steps += 1
        if high_level_policy_update:
            self.hl_steps +=1

        # Decide if we should update the belief
        high_level_belief_update = self.steps % self.lh_ratio == 0

        # Save previous target velocity, used for target dynamics computation
        self.old_targets_vel = np.copy(self.targets_vel)

        # Move the targets
        if not envtest:
            self.move_targets(self.delta_t, dyn_sigma=self.dyn_sigma)  # move targets according to dynamic model "self.env_mode"

        last_location = np.copy(self.robots)

        # We apply the action to robot 0, moving it to its current chosen location in the neighborhood if possible
        for r in range(self.nrobots):
            # map actions to actual action space.
            # Map the action space to the circle of unit radius
            #if np.linalg.norm(action[r]) > 1: action[r] = action[r] / np.linalg.norm(action[r])
            # map actions to actual action space.
            action[r] = action[r] * self.action_space_mapping

            location = np.copy(self.robots[r])

            if not self.realistic_Dengine:
                if self.env_mode == 'cte_vel' or self.env_mode=='sf_goal' or self.env_mode=='brown' or self.env_mode=='static':
                    #applied_action = np.clip(action, self.action_space.low*self.delta_t, self.action_space.high*self.delta_t)
                    applied_action=action[r]*self.delta_t
                else:
                    applied_action = action[r]

                applied_rot = applied_action[2]
                location[0:2] += self.rotate_action_pos(applied_action[0:2],location[2]) #action[0:2]
                location[2] = (location[2] + 60 * applied_action[2]) % 360

            else: # INSERT REAL DYNAMICS
                action_global_frame_2d = self.rotate_action_pos(action[r][0:2],location[2])
                self.drone_engines[r].step(np.array([action_global_frame_2d]))

                applied_rot = action[r][2] * self.delta_t
                location[0:2] = self.drone_engines[r]._getDronePos()[0]
                location[2] = (location[2] + 60 * applied_rot) % 360

            self.place_robot(r, location)
            self.last_action[r][0:2] = self.rotate_action_pos(self.last_action[r][0:2],-applied_rot*60)

        # Now all positions from robots and targets are updated. Next: obtain new observation
        # Initialize entropy measurements (for observation and rewards) and logging variables
        num_targets_tracked = 0
        entropy_measurements = 0
        entropy_beliefs = 0
        new_targets_tracked = 0

        # Check vis   ibility of targets, reused when obtaining probability estimates on target classes
        self.visibility = self.check_visibility()

        # We obtain probability estimates and update beliefs on target class for all the targets
        for t in range(self.ntargets):
            new_belief = self.beliefs[0][t, :].copy()
            for r in range(self.nrobots):
                # We take probability estimates of the target t from the new location
                #if high_level_belief_update:
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
                        new_targets_tracked += 1
                    num_targets_tracked += 1

        #print("visibility:", self.visibility)
        #print("new observation", self._observation)
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
                self.episode_results = self.log_performance_episode()
        else:
            # Constant penalty for processing a new measurement
            if high_level_belief_update: reward += -0.3
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
        else:
            if self.multiagent_policy:
                returned_obs = [[obs[i]["targets"]["all"], obs[i]["robots"]] for i in
                                range(self.nrobots)]  # apparently this works better in our old policies
                # returned_obs = [[obs[i]["targets"]["all"], []] for i in range(self.nrobots)] # this is only temporary TODO TAKE OUT AND UNCOMMENT ABOVE
            else:
                returned_obs = [obs[i]["targets"]["all"] for i in range(self.nrobots)]
            return returned_obs


    def observation(self, target, robot):
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
            foldername = foldername + '_' + str(SIDE)
        if self.realistic_Dengine:
            foldername = foldername + '_realistic'
        if self.lh_ratio!=1:
            foldername = foldername + '_' + str(self.lh_ratio)
        if self.number_of_classes!=2:
            foldername = foldername + '_' + str(self.number_of_classes)+'classes'
        if self.dyn_sigma!=0:
            foldername = foldername + '_dynSigma' + str(self.dyn_sigma)
        if self.simulated_perception != "dummy":
            foldername = foldername + '_per_' + str(self.dyn_sigma)
        if self.multiagent_policy:
            foldername = foldername + '_MA'


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
        return csvDict

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen. If established a priori, saves the generated images.
        """
        # print current step and beliefs
        print(f'Step: {self.steps}')
        print(f'Belief: {self.beliefs}')

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
            self.plot_image = True
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

        # Only render if we are testing the policy
        if self.test:
            if self.save_video and self.steps>1:
                if self.plot_image: self.fig3.savefig(self.video_foldername+'camera/'+str(self.steps)+'.png')
                if self.plot_scene: self.fig.savefig(self.video_foldername + 'state/' + str(self.steps) + '.png')
                if self.plot_beliefs: self.fig2.savefig(self.video_foldername + 'beliefs/' + str(self.steps) + '.png')




if __name__ == "__main__":
    """
    Used for testing the environment
    """
    config = {
        "dimensions": 3,
        "ntargets": [12],
        "nrobots": [1],
        "nclasses": 2,
        "MAXTARGETS":10,
        "horizon": 1000,
        "env_mode": "sf_goal",
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

    plot_image = False
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
        action = {'0':np.array([1, 0, 0])}#,'1':np.array([0, 0, 0.25])}#np.random.uniform(-1, 1, 3)
        #action[0] = 0
        #action[2] = 0
        obs, reward, final, _ = sc.step(action) #, envtest = True)
        print(obs)
        print(sc.beliefs[0])
        print(sc._observation)
        print(reward)
        print("class:", sc.targets_ID[0])
        # plt.imshow(sc.unrealPedestrianData[sc.targets_ID[0]].getObservation(sc.relative_pose(0,0)))

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



