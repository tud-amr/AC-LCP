
from abc import ABC

import ray
import numpy as np

from ray.rllib import Policy
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.util.iter import LocalIterator
from ray.rllib.execution.rollout_ops import ParallelRollouts, SelectExperiences
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents import with_common_config
DEFAULT_CONFIG = with_common_config({})


TrainerConfigDict = dict

class HeuristicHandcraftedPolicy(Policy):
    """Example of a custom policy written from scratch.
    You might find it more convenient to extend TF/TorchPolicy instead
    for a real policy.
    """

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.config["framework"] = None
        # example parameter
        self.w = 1.0
        self.ntargets = config['env_config']['MAXTARGETS']
        self.child_space = 8

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return random actions
        #print(obs_batch[0][3:3+3])
        def guidance_policy(obs):
            # print(obs)
            #observation_all = np.reshape(obs,(3,self.child_space*self.ntargets+1))[-1]
            #targetpose = np.reshape(obs[1:], (self.ntargets,self.child_space))[0][1:1+3]
            targetpose = np.reshape(obs[1:], (self.ntargets, self.child_space))[0][1:1 + 3]
            xaction = np.clip(targetpose[0]-3.375,self.action_space.low[0],self.action_space.high[0])
            yaction = np.clip(targetpose[1], self.action_space.low[1], self.action_space.high[1])
            angle = (targetpose[2]-np.pi)/(np.pi/3) if targetpose[2]>=0 else (np.pi+targetpose[2])/(np.pi/3)

            #print('relative angle to confront:', angle*180/np.pi)

            theta_action = np.clip(angle, self.action_space.low[2], self.action_space.high[2])
            return np.array([xaction,yaction,theta_action])

        #observation_batch = obs_batch[:][-8:]
        return np.array(
            [guidance_policy(obs) for obs in obs_batch]), [], {}

    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}

    def update_some_value(self, w):
        # can also call other methods on policies
        self.w = w

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]


def execution_plan(workers: WorkerSet,
                   config: TrainerConfigDict) -> LocalIterator[dict]:
    rollouts = ParallelRollouts(workers, mode="async")

    # Collect batches for the trainable policies.
    rollouts = rollouts.for_each(
        SelectExperiences(workers.trainable_policies()))

    # Return training metrics.
    return StandardMetricsReporting(rollouts, workers, config)


HeuristicHandcraftedTrainer = build_trainer(
    name="HeuristicHandcraftedTrainer",
    default_config=DEFAULT_CONFIG,
    default_policy=HeuristicHandcraftedPolicy,
    execution_plan=execution_plan)