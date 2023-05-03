from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy

from typing import Dict
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
import time
import csv
import os


def trainEvlogs2csv(folder_name,file_name,csvDict, n_episode):
    fieldnames = list(csvDict.keys())
    if n_episode == 1:
        #print('this happens')
        csvfile = open(folder_name + file_name, 'w', newline='')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    else:
        csvfile = open(folder_name + file_name, 'a', newline='')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writerow(csvDict)


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):  # , env_index: int, **kwargs):
        # print("episode {} (env-idx) started.".format(episode.episode_id))
        episode.user_data["step"] = 0
        #episode.user_data["nb_communications_episode"] = 0
        #episode.user_data["collisions"] = 0
        #episode.user_data["list_communications"] = []
        #episode.user_data["comms_per_step"] = []
        #episode.user_data["histogram_communications"] = np.zeros(policies["policy_0"].config["horizon"]+1)#np.zeros(101)
        episode.user_data["secs_per_episode"] = time.time()
        episode.user_data["mean_secs_per_ts"] = 0
        episode.user_data["auxiliary_time"] = time.time()
        episode.user_data["auxiliary_time_episode"] = time.time()

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        auxtime_ts = time.time() - episode.user_data["auxiliary_time"]
        episode.user_data["mean_secs_per_ts"] += auxtime_ts
        aux = episode.last_observation_for(0)
        aux12 = aux.reshape((1,aux.shape[0]))
        #aux2 = episode.last_action_for(0)
        #aux3 = episode.last_info_for(0)
        n_agents = worker.env.num_agents
        episode.batch_builder.count += n_agents - 1  ### THIS ONE IS THE ONE TO CHANGE!!
        #collisions = 0
        if episode.user_data["step"] != 0:
            for i in range(n_agents):
                aux4 = episode.policy_for(i)
                #episode.user_data["nb_communications_episode"] += episode.last_info_for(i)["communication"]
                #collisions += episode.last_info_for(i)["collisions"]
                #for ii in range(int(episode.last_info_for(i)["communication"])):
                #    episode.user_data["comms_per_step"].append(episode.user_data["step"])
                #episode.user_data["histogram_communications"][episode.user_data["step"]] += episode.last_info_for(i)["communication"]

            #episode.user_data["list_communications"].append(episode.user_data["nb_communications_episode"])
            #episode.user_data["collisions"] += collisions / 2
            #if episode.user_data["collisions"] != 0:
             #   print("COLLISION!")
        episode.user_data["step"] += 1
        #print(episode.user_data["step"])
        debug = False
        if debug:
            workerdebug = worker
            policy_index = worker.policies_to_train[0]
            policydebug = worker.policy_map[policy_index]
            wdebug = policydebug.get_weights()
            predicate = policydebug.loss_initialized()
            if predicate:
                # overwrite default VF prediction with the central VF
                #sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                #sample_batch[SampleBatch.CUR_OBS], sample_batch[OPPONENT_OBS],
                #sample_batch[OPPONENT_ACTION])
                t1 = time.time()
                #encoder_debug = policydebug.compute_encoding_layer(aux12)
                #print(encoder_debug)
                t1 = time.time()-t1
                t2 = time.time()
                action_debug = policydebug.compute_action(aux12)
                output_inputs = policydebug.output_inputs(aux12)
                t2 = time.time() - t2
            #print("WEIGHTS")
            #print(wdebug)
            #print("END WEIGHTS")
            episode.user_data["auxiliary_time"] = time.time()



    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode, **kwargs):
        #horizon = policies["policy_0"].config["horizon"]
        #print("episode time:",time.time() - episode.user_data["auxiliary_time_episode"])
        policy_index = worker.policies_to_train[0]
        episode.custom_metrics["secs_per_episode"] = time.time()-episode.user_data["secs_per_episode"]
        #episode.custom_metrics["mean_secs_per_ts"] = episode.user_data["mean_secs_per_ts"] / policies["policy_0"].config["horizon"]  #100.0
        episode.custom_metrics["mean_secs_per_ts"] = episode.user_data["mean_secs_per_ts"] / \
                                                     policies[policy_index].config["horizon"]  # 100.0
        episode.custom_metrics["nb targets tracked"] = np.sum(worker.env.num_targets_tracked[0])
        episode.custom_metrics["episode length"] = worker.env.steps
        episode.custom_metrics["success_rate"] = worker.env.success
        episode.custom_metrics["entropy"] = worker.env.last_entropy_log
        if worker.env.ntargets == 3:
            episode.custom_metrics["nb_targets_tracked_3targets"] = np.sum(worker.env.num_targets_tracked[0])
            episode.custom_metrics["episode length_3targets"] = worker.env.steps
        if worker.env.ntargets == 6:
            episode.custom_metrics["nb_targets_tracked_6targets"] = np.sum(worker.env.num_targets_tracked[0])
            episode.custom_metrics["episode length_6targets"] = worker.env.steps
        if worker.env.ntargets == 9:
            episode.custom_metrics["nb_targets_tracked_9targets"] = np.sum(worker.env.num_targets_tracked[0])
            episode.custom_metrics["episode length_9targets"] = worker.env.steps
        if worker.env.ntargets == 12:
            episode.custom_metrics["nb_targets_tracked_12targets"] = np.sum(worker.env.num_targets_tracked[0])
            episode.custom_metrics["episode length_12targets"] = worker.env.steps

        #episode.custom_metrics["nb_communications_episode"] = episode.user_data["nb_communications_episode"]
        #episode.custom_metrics["collisions"] = episode.user_data["collisions"]
        #episode.hist_data["communications_histogram"] = episode.user_data["list_communications"]
        #episode.hist_data["communications_per_step"] = episode.user_data["comms_per_step"]
        #if episode.user_data["collisions"] == 0:
        #    episode.custom_metrics["success"] = 1
        #else:
        #    episode.custom_metrics["success"] = 0

        #episode.custom_metrics["scn_"+str(base_env.envs[0]._env.world.current_scenario)+"_success"] = episode.custom_metrics["success"]
        #episode.custom_metrics["scn_" + str(base_env.envs[0]._env.world.current_scenario) + "_nb_comm_episode"] = episode.custom_metrics["nb_communications_episode"]
        #nEpisodesxScenario = worker.policy_config['train_batch_size']/(base_env.envs[0]._env.num_agents*policies["policy_0"].config["horizon"])
        #if base_env.envs[0]._env.episode_id % nEpisodesxScenario == 0 and base_env.envs[0]._env.world.test == 1:  # Alternatively use the in_evaluation value in config
        #if base_env.envs[0]._env.world.time_step % (worker.policy_config['train_batch_size']/base_env.envs[0]._env.num_agents) == 0 and worker.policy_config["model"]["custom_model_config"]["training"]==False:  # Alternatively use the in_evaluation value in config
        #if base_env.envs[0]._env.world.time_step % (worker.policy_config['train_batch_size']) == 0 and worker.policy_config["model"]["custom_model_config"]["training"] == False:  # Alternatively use the in_evaluation value in config
            #base_env.envs[0]._env.world.next_eval_scenario()
            #base_env.envs[0]._env.world.set_eval_scenario()

        # training_eval_logs = base_env.envs[0]._env.world.test
        # if worker.policy_config["model"]["custom_model_config"]["training"]==False and training_eval_logs == True:
        #     goal_achieved = 1
        #     for i in range(worker.env.num_agents):
        #         if episode.last_info_for(i)["goal_achieved"] == 0:
        #             goal_achieved = 0
        #
        #     if episode.custom_metrics["success"] == 0:
        #         goal_achieved = 0
        #
        #     timesteps = episode.last_info_for(0)["step"]
        #
        #     checkpoint = policies["policy_0"].config["model"]["custom_model_config"]["checkpoint"]
        #     scenario = base_env.envs[0]._env.world.current_scenario
        #     folder_name = base_env.envs[0]._env.folder_name + 'training_eval/'
        #     file_name = 'checkpoint_' + str(checkpoint) + '.csv'
        #     n_episode = base_env.envs[0]._env.episode_id
        #     csvDict = {
        #         'scenario': scenario,
        #         'ep_safety': episode.custom_metrics["success"],
        #         'ep_nb_communications': episode.custom_metrics["nb_communications_episode"],
        #         'ep_goal_achieved': goal_achieved,
        #         'step': timesteps
        #     }
        #     for i in range(policies["policy_0"].config["horizon"]+1):  #101):
        #         csvDict[str(i)] = episode.user_data["histogram_communications"][i]
        #
        #     if not os.path.isdir(folder_name):
        #         os.makedirs(folder_name)
        #     trainEvlogs2csv(folder_name,file_name,csvDict, n_episode)


        #print("episode END")

    #"""
    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        print("returned sample batch of size {}".format(samples.count))
    #"""

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        #print("trainer.train() result: {} -> {} episodes".format(trainer, result["episodes_this_iter"]))
        work = trainer.workers.local_worker()
        policy_index = work.policies_to_train[0]
        policydebug = work.policy_map[policy_index]
        wdebug = policydebug.get_weights()
        #kernelweights = wdebug['default_policy/dense/kernel']
        #if np.any(np.isnan(kernelweights)):
        #    print("here's a nan")
        # if trainer.config["model"]["custom_model_config"]["training"]:
        #     if 313 <= trainer.iteration < 313 * 2:  # change distribution of sampled episodes 15000 episodes
        #         work.foreach_env(lambda env: env._env.world.set_scenario_distr([0.25, 0.75, 0]))
        #     if 313 * 2 <= trainer.iteration < 313 * 3:  # change distribution of sampled episodes 15000 episodes
        #         work.foreach_env(lambda env: env._env.world.set_scenario_distr([0.125, 0.125, 0.75]))
        #     if 313 * 3 <= trainer.iteration < 313 * 4:  # change distribution of sampled episodes 15000 episodes
        #         work.foreach_env(lambda env: env._env.world.set_scenario_distr([0.0, 0.25, 0.75]))
        #     if 313 * 4 <= trainer.iteration:  # change distribution of sampled episodes 15000 episodes
        #         work.foreach_env(lambda env: env._env.world.set_scenario_distr([0.0, 0.0, 1.0]))


        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True
    """
    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
    #"""