import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
import os
import pandas as pd

def dataextract(episodes, ntargets_list, foldernames, relative_to_heuristic):
    info_model = []
    bdataframe = pd.read_csv(foldernames[0], header= None)
    for i in range(0,len(foldernames)):
        dataframe = pd.read_csv(foldernames[i], header= None)
        auxDict = {'ntargets':[],'episode_length_mean':[],'episode_length_std':[],'success':[],'ntracked_mean':[],'ntracked_std':[],'ntracked_min':[],'ntracked_max':[],
                   'entropy_mean':[],'entropy_std':[],'entropy_min':[],'entropy_max':[],} #'success_std':[]}
        for numtarget in ntargets_list:
            auxDict['ntargets'].append(numtarget)
            #relSuccessData = dataframe[dataframe[2]==numtarget][4]
            relSuccessData = dataframe[dataframe[2] == numtarget][5]==numtarget
            if relative_to_heuristic:
                relSuccessData -= bdataframe[dataframe[2]==numtarget][4]
            auxDict['success'].append(relSuccessData.mean())
            #relStepData = dataframe[dataframe[2] == numtarget][dataframe[4]==1][3]
            relStepData = dataframe[dataframe[2] == numtarget][dataframe[5] == numtarget][3]
            if relative_to_heuristic:
                relStepData -= bdataframe[dataframe[2] == numtarget][dataframe[4]==1][3]
            auxDict['episode_length_mean'].append(relStepData.mean())
            auxDict['episode_length_std'].append(relStepData.std())
            relTrackedData = dataframe[dataframe[2] == numtarget][5]
            if relative_to_heuristic:
                relTrackedData -= bdataframe[dataframe[2] == numtarget][5]
            relTrackedData = relTrackedData / numtarget *100
            auxDict['ntracked_mean'].append(relTrackedData.mean())
            auxDict['ntracked_std'].append(relTrackedData.std())
            auxDict['ntracked_min'].append(relTrackedData.min())
            auxDict['ntracked_max'].append(relTrackedData.max())
            relEntropyData = dataframe[dataframe[2] == numtarget][6]
            if relative_to_heuristic:
                relEntropyData -= bdataframe[dataframe[2] == numtarget][6]
            auxDict['entropy_mean'].append(relEntropyData.mean())
            auxDict['entropy_std'].append(relEntropyData.std())
            auxDict['entropy_min'].append(relEntropyData.min())
            auxDict['entropy_max'].append(relEntropyData.max())

        info_model.append(auxDict)
    return info_model

def extracthistograms(episodes, ntargets_list, foldernames, relative_to_heuristic,seeds,horizon):
    info_model = []
    bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    bdataframe = pd.read_csv(foldernames[0], header=None)
    for i in range(0, len(foldernames)):
        dataframe = pd.read_csv(foldernames[i], header=None)
        dataframe = dataframe[dataframe[2] == ntargets_list[0]]

        obsed = list(dataframe[8])
        obsed_tracked = list(dataframe[9])
        tracked = list(dataframe[10])
        tobs_list = [int(a) for a in (','.join(obsed)).split(',')]
        tobs_hist = np.histogram(tobs_list,bins=bins)
        tobs_hist = tobs_hist[0]/50
        tobs_episode = [[int(i) for i in a.split(',')] for a in obsed]
        for llist in tobs_episode:
            if len(llist) < horizon:
                llist += [0]*(horizon-len(llist))
        tobs_episode = np.array([np.array(episode) for episode in tobs_episode])
        tobs_episode = tobs_episode.mean(axis=0)

        tobs_episode_tracked = [[int(i) for i in a.split(',')] for a in obsed_tracked]
        for llist in tobs_episode_tracked:
            if len(llist) < horizon:
                llist += [0]*(horizon-len(llist))
        tobs_episode_tracked = np.array(tobs_episode_tracked).mean(axis=0)

        episode_tracked = [[int(i) for i in a.split(',')] for a in tracked]
        for llist in episode_tracked:
            if len(llist) < horizon:
                llist += [llist[-1]] * (horizon - len(llist))
        episode_tracked = np.array(episode_tracked).mean(axis=0)*100/ntargets_list[0]

        tobs_episode_nottracked = tobs_episode-tobs_episode_tracked
        info_model.append((tobs_hist,tobs_episode,tobs_episode_tracked, tobs_episode_nottracked, episode_tracked))

    ed_idx = 0
    hist_mean = []
    hist_std = []
    hist_ep_mean = []
    hist_ep_std = []
    hist_ep_tracked_mean = []
    hist_ep_tracked_std = []
    hist_ep_nottracked_mean = []
    hist_ep_nottracked_std = []
    hist_tracked_mean = []
    hist_tracked_std = []
    for nseeds in seeds:
        tobs_hist_aux = []
        tobs_episode_aux = []
        tobs_episode_tracked_aux = []
        tobs_episode_nottracked_aux = []
        episode_tracked_aux = []
        for i in range(nseeds):
            tobs_hist_aux.append(info_model[ed_idx][0])
            tobs_episode_aux.append(info_model[ed_idx][1])
            tobs_episode_tracked_aux.append(info_model[ed_idx][2])
            tobs_episode_nottracked_aux.append(info_model[ed_idx][3])
            episode_tracked_aux.append(info_model[ed_idx][4])

            ed_idx += 1
        hist_mean.append(np.array(tobs_hist_aux).mean(axis=0))
        hist_std.append(np.array(tobs_hist_aux).std(axis=0))
        hist_ep_mean.append(np.array(tobs_episode_aux).mean(axis=0))
        hist_ep_std.append(np.array(tobs_episode_aux).std(axis=0))
        hist_ep_tracked_mean.append(np.array(tobs_episode_tracked_aux).mean(axis=0))
        hist_ep_tracked_std.append(np.array(tobs_episode_tracked_aux).std(axis=0))
        hist_ep_nottracked_mean.append(np.array(tobs_episode_nottracked_aux).mean(axis=0))
        hist_ep_nottracked_std.append(np.array(tobs_episode_nottracked_aux).std(axis=0))
        hist_tracked_mean.append(np.array(episode_tracked_aux).mean(axis=0))
        hist_tracked_std.append(np.array(episode_tracked_aux).std(axis=0))


    return hist_mean,hist_std,hist_ep_mean,hist_ep_std,hist_ep_tracked_mean,hist_ep_tracked_std,hist_ep_nottracked_mean,hist_ep_nottracked_std,hist_tracked_mean,hist_tracked_std

def compress_seed_data(extracted_data,ed_idx,nseeds):
    important_data = extracted_data[ed_idx:ed_idx+nseeds]
    auxDict = {'ntargets': [], 'episode_length_mean': [], 'episode_length_std': [], 'success': [], 'ntracked_mean': [],
               'ntracked_std': [], 'ntracked_min': [], 'ntracked_max': [],
               'entropy_mean': [], 'entropy_std': [], 'entropy_min': [], 'entropy_max': [], }  # 'success_std':[]}
    for key in auxDict:
        for i in range(nseeds):
            if key=='ntargets' and i>0: break
            auxDict[key].append(important_data[i][key])
        if key!='ntargets':
            auxDict[key] = np.array(auxDict[key])

    auxDict['ntargets'] = auxDict['ntargets'][0]
    auxvar = auxDict['episode_length_mean'].copy()
    auxDict['episode_length_mean'] = list(auxvar.mean(axis=0))
    auxDict['episode_length_std'] = list(auxvar.std(axis=0))
    auxDict['success'] = list(auxDict['success'].mean(axis=0))
    auxvar = auxDict['ntracked_mean'].copy()
    auxDict['ntracked_mean'] = list(auxvar.mean(axis=0))
    auxDict['ntracked_std'] = list(auxvar.std(axis=0))
    auxDict['ntracked_min'] = list(auxvar.min(axis=0))
    auxDict['ntracked_max'] = list(auxvar.max(axis=0))
    auxvar = auxDict['entropy_mean'].copy()
    auxDict['entropy_mean'] = list(auxvar.mean(axis=0))
    auxDict['entropy_std'] = list(auxvar.std(axis=0))
    auxDict['entropy_min'] = list(auxvar.min(axis=0))
    auxDict['entropy_max'] = list(auxvar.max(axis=0))

    return auxDict





if __name__ == '__main__':

    ## Loading the files

    visibility_hist = True
    #print(rolloutArgs.checkpoint)
    #### in code trials
    #args.run = 'MixinPPO'
    episodes = 50
    ntargets_list = [30]
    #env = 'SceneEnv_v3'
    relative_to_heuristic = False
    ####
    #policy_dir = 'fully_connected_max10targets_longtrain'
    #policy_dir = 'transformer_stable_nodropout_largetrain'
    #policy_dir = 'deepsets_largetrain_1to3agents'
    #policy_dir = 'deepsets_largetrain_1to3agents_restored'

    # policy_dir = ["heuristic_policy_config","baseline_2_3_upto_3_targets"]+["policy_versions/attentionVsDeepSets/cte_vel/deep_sets",
    #                                                                         "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/seed100/1_12_targets/low_ntargets",
    #                                                                         "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/seed200/1_12_targets/low_ntargets"]
    #policy_dir = ["transf_st_drop_1layer_1target","transf_st_drop_1layer_1target", "transf_st_drop_1layer_1target",
                  #"transf_st_drop_1layer_1target_static_6t","transf_st_drop_1layer_1target_static_8t"]#["transf_st_drop_1layer_1target",
                  #"transf_st_drop_1layer_1target","transf_st_drop_1layer_1target","transf_st_drop_1layer_1target"]
    # policy_dir = [#"heuristic_policy_config",
    #               #"policy_versions/baseline2_static/seed100", "policy_versions/baseline2_static/seed200", "policy_versions/baseline2_static/seed300",
    #               #"policy_versions/baseline2/seed100", "policy_versions/baseline2/seed200", "policy_versions/baseline2/seed300",
    #               #"policy_versions/attentionVsDeepSets/cte_vel/deep_sets/lstm/1_12_targets/seed100",
    #               #"policy_versions/attentionVsDeepSets/cte_vel/deep_sets/lstm/1_12_targets/seed200",
    #               #"policy_versions/attentionVsDeepSets/cte_vel/deep_sets/lstm/1_12_targets/seed300",
    #               #"policy_versions/attentionVsDeepSets/cte_vel/deep_sets/lstm/1_12_targets/low_ntargets/seed100",
    #               #"policy_versions/attentionVsDeepSets/cte_vel/deep_sets/lstm/1_12_targets/low_ntargets/seed200",
    #               #"policy_versions/attentionVsDeepSets/cte_vel/deep_sets/lstm/1_12_targets/low_ntargets/seed300",
    #               ] + [
    #     #"policy_versions/attentionVsDeepSets/cte_vel/deep_sets/attention_deepsets_fairparam/1_12_targets/low_ntargets/seed100",
    #     #"policy_versions/attentionVsDeepSets/cte_vel/deep_sets/attention_deepsets/1_12_targets/low_ntargets/seed100",
    #     #"policy_versions/attentionVsDeepSets/cte_vel/deep_sets/attention_deepsets/1_12_targets/low_ntargets/seed200",
    #     #"policy_versions/attentionVsDeepSets/cte_vel/deep_sets/attention_deepsets/1_12_targets/low_ntargets/seed300",
    #     #"policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/seed100/1_12_targets/low_ntargets",
    #     #"policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/seed200/1_12_targets/low_ntargets",
    #     #"policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/seed300/1_12_targets/low_ntargets",
    #     ]
    # ncheckpoint = [#1,
    #                #1000, 1000, 1000,
    #                #500, 500, 500,
    #                #6000,6000,6000,
    #                #11000,11000,11000
    #                ]+[#11000,
    #                   #11000,11000,11000,
    #                   #10710,11000,11000
    #                 ]#[3000, 6000, 10710]#[800,800]#[1090, 1090]#[950,950,1050,450,400] #[950,950,950,950] #460
    # experiment_list = [#1,
    #                    #1,1,1,
    #                    #1,1,1,
    #                    #4,3,3,
    #                    #1,1,1
    #                    ]+[#1,
    #                       #1,1,1,
    #                      # 1,1,1
    # ]#[19, 4, 1] #[1,1,2,1,1]#[1,1,1,1] #3#3  # 1
    #
    # #n_testmodels = 3#7
    # heuristic_handcrafted = [#True,
    #                          #False, False, False,
    #                          #False, False, False,
    #                          #False, False, False,
    #                          #False, False, False
    #                          ]#+[False]*n_testmodels #+[False, False]
    # heuristic_policy = [#True,
    #                     #True, True, True,
    #                     #True, True, True,
    #                     #False, False, False,
    #                     #False, False, False
    #                     ]#+[False] * n_testmodels #[False,True, False,False,False]#[False,True, False, True]
    # heuristic_target_order = [#True,
    #                           #True, True, True,
    #                           #True, True, True,
    #                           #True, True, True,
    #                           #True, True, True
    #                           ]# + [False]*n_testmodels
    # seeds = [#1,
    #          #3,
    #          #3,
    #          #3,
    #          #3,
    #          ]# + [#1,
    #                #3,
    #                3]
    # #static_env = [True, True,True]#[True,True, True,True,True] #[False, False, True, True]
    # env_mode = ['sf_goals']*(n_testmodels+13) #+['sf_goal']*2#['static','brown','cte_vel','sf_goal']
    # #env_mode_name = ['baseline 1','baseline 2 seed 1', 'baseline 2 seed 2','baseline 2 seed 3','seed 1','seed 2','seed 3']#+['baseline 2', 'baseline 3 (wip)']
    # env_mode_name = ['Hand-crafted',
    #                  #'baseline 2 trained static',
    #                  'Single-target',
    #                  #'LSTM',
    #                  #'LSTM refined',
    #                  ]+\
    #                 [#'Attention Deep Sets Large',
    #                  #'Attention Deep Sets Best',
    #                  'Ours']  # +['baseline 2', 'baseline 3 (wip)']





    #
    #
    #
    # horizon = 500
    #
    # folder_policy_list = []
    #
    # folder_dict = {}
    # ###
    #
    # for i in range(len(policy_dir)):
    #     # Fixing save-dir mechanic to make it simpler
    #     savedir = './ray_results'
    #     if policy_dir[i] != '':
    #         savedir = savedir + '/' + policy_dir[i]
    #     else:
    #         savedir = savedir + '/' + env
    #     # if not os.path.isdir(savedir):
    #     #    os.makedirs(savedir)
    #     experiment = experiment_list[i]
    #     expstr = '/exp' + str(experiment)
    #     video_dir = savedir + '/videos_exp_' + str(experiment) + '_ckpoint_' + str(ncheckpoint[i])+'/'
    #     if heuristic_handcrafted[i]: video_dir = savedir + '/videos_exp_'+str(experiment)+'/'
    #     folder_name = video_dir + 'test_performance'
    #     if heuristic_policy[i]:
    #         folder_name = folder_name + '_heuristic'
    #     if heuristic_target_order[i]:
    #         folder_name = folder_name + '_heuristicTargetOrder'
    #     if horizon!=100:
    #         folder_name = folder_name + '_' + str(horizon)
    #     if visibility_hist:
    #         folder_name = folder_name + '_histogram'
    #
    #     folder_name = folder_name + '_' + env_mode[i]
    #     folder_policy_list.append(folder_name + '.csv')
    policy_dir = [  # "heuristic_policy_config",
        # "policy_versions/baseline2/seed100",
        # "policy_versions/baseline2/seed200",
        # "policy_versions/baseline2/seed300",
        # "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/seed100/1_12_targets/low_ntargets",
        # "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/seed200/1_12_targets/low_ntargets",
        # "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/seed300/1_12_targets/low_ntargets",
        # "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/50x50_env/seed100/1_12_targets",
        # "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/50x50_env/seed100/2ndphase",
        # "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/50x50_env/seed100/2ndphase",
        # "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2_2/50x50_env/seed100/1_12_targets",  # BUG IN ENVIRONMENT
        # "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v3/50x50_env/seed100/1_12_targets",  # BUG IN ENVIRONMENT
        # "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v3/50x50_env/seed100/2ndphase"  # BUG IN ENVIRONMENT
    ]
    ncheckpoint = [  # 1,
        # 6000, 6000, 6000,
        # 10710, 11000, 11000,
        # 6000,
        # 8030,
        # 8030,
        # 4260,
        # 5840,
        # 7400
    ]  # [3000, 6000, 10710]#[800,800]#[1090, 1090]#[950,950,1050,450,400] #[950,950,950,950] #460
    experiment_list = [  # 1,
        # 1, 1, 1,
        # 1, 1, 1,
        # 4,
        # 1,
        # 1,
        # 3,
        # 4,
        # 1
    ]  # [19, 4, 1] #[1,1,2,1,1]#[1,1,1,1] #3#3  # 1

    n_testmodels = 0
    heuristic_handcrafted = [  # True,
        # False, False, False,
        # False, False, False,
        # False,
        # False,
        # False,
        # False,
        # False,
        # False
    ]
    heuristic_policy = [  # True,
        # True, True, True,
        # False, False, False,
        # False,
        # False,
        # False,
        # False,
        # False,
        # False
    ]
    heuristic_target_order = [  # True,
        # True, True, True,
        # False, False, False,
        # False,
        # False,
        # False,
        # False,
        # False,
        # False
    ]
    robot_robot_occlusion = True
    robot_target_assignment = [  # True,
        # True, True, True,
        # False, False, False,
        # False,
        # False,
        # False,
        # False,
        # False,
        # False
    ]
    realistic_Dengine = [  # False,
        # False, False, False,
        # False, False, False,
        # False,
        # False,
        # True,
        # False,
        # False,
        # False
    ]

    lh_ratio = [  # 1,
        # 1, 1, 1,
        # 1, 1, 1,
        # 1,
        # 1,
        # 5,
        # 1,
        # 1,
        # 1
    ]

    seeds = [  # 1, #3, 3,1,1,1,1,1,1
    ]
    # static_env = [True, True,True]#[True,True, True,True,True] #[False, False, True, True]
    # env_mode_name = ['baseline 1','baseline 2 seed 1', 'baseline 2 seed 2','baseline 2 seed 3','seed 1','seed 2','seed 3']#+['baseline 2', 'baseline 3 (wip)']
    env_mode_name = [  # 'baseline 1 - TA',# 'baseline 2 - TA', 'Ours 16x16', 'Ours 50x50 1st phase',
        # 'Ours50x50 2ndPhase (inc)','Ours50x50 2ndPhase(inc)-RD','Ours v2 50x50 1stPhase(inc)','Ours 50x50 2arch-1(inc)', 'Ours 50x50  2arch-2(inc)'
    ]  # +['baseline 2', 'baseline 3 (wip)']

    plot_curve = []

    # RSS paper
    addendum = [
        "Ours/setTransformer_opt_v2/50x50_env/seed100/2ndphase",
        "Ours/setTransformer_opt_v2/50x50_env/seed200/2ndphase",
        "Ours/setTransformer_opt_v2/50x50_env/seed300/2ndphase",
        "Ours/setTransformer_opt_v2/50x50_env/seed400/2ndphase",
        "Ours/setTransformer_opt_v2/50x50_env/seed500/2ndphase"
    ]
    policy_dir += addendum
    ncheckpoint += [8000] * len(addendum)
    experiment_list += [1] * len(addendum)
    heuristic_handcrafted += [False] * len(addendum)
    heuristic_policy += [False] * len(addendum)
    heuristic_target_order += [False] * len(addendum)
    robot_target_assignment += [False] * len(addendum)
    realistic_Dengine += [True] * len(addendum)
    lh_ratio += [5] * len(addendum)
    seeds += [len(addendum)] * int(len(addendum) != 0)
    env_mode_name += ['Ours'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)

    # LSTM baseline
    addendum = [
        "LSTM/seed100/2ndphase",
        "LSTM/seed200/2ndphase",
        "LSTM/seed300/2ndphase",
        "LSTM/seed400/2ndphase",
        "LSTM/seed500/2ndphase"
    ]
    policy_dir += addendum
    ncheckpoint += [8000] * len(addendum)
    experiment_list += [1] * len(addendum)
    heuristic_handcrafted += [False] * len(addendum)
    heuristic_policy += [False] * len(addendum)
    heuristic_target_order += [True] * len(addendum)
    robot_target_assignment += [True] * len(addendum)
    realistic_Dengine += [True] * len(addendum)
    lh_ratio += [5] * len(addendum)
    seeds += [len(addendum)] * int(len(addendum) != 0)
    env_mode_name += ['LSTM encoder'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)

    # Single-target baseline
    addendum = [
        "baseline_2/seed100",
        "baseline_2/seed200",
        "baseline_2/seed300",
        "baseline_2/seed400",
        "baseline_2/seed500",
    ]
    policy_dir += addendum
    ncheckpoint += [6000] * len(addendum)
    experiment_list += [1,1,1,3,5]
    heuristic_handcrafted += [False] * len(addendum)
    heuristic_policy += [True] * len(addendum)
    heuristic_target_order += [True] * len(addendum)
    robot_target_assignment += [True] * len(addendum)
    realistic_Dengine += [True] * len(addendum)
    lh_ratio += [5] * len(addendum)
    seeds += [len(addendum)] * int(len(addendum) != 0)
    env_mode_name += ['Single-target'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)

    # Hard-coded baseline
    addendum = [
        "baseline_1",
    ]
    policy_dir += addendum
    ncheckpoint += [6000] * len(addendum)
    experiment_list += [1] * len(addendum)
    heuristic_handcrafted += [True] * len(addendum)
    heuristic_policy += [True] * len(addendum)
    heuristic_target_order += [True] * len(addendum)
    robot_target_assignment += [True] * len(addendum)
    realistic_Dengine += [True] * len(addendum)
    lh_ratio += [5] * len(addendum)
    seeds += [len(addendum)] * int(len(addendum) != 0)
    env_mode_name += ['Hand-crafted'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)

    env_mode = ["static"] * np.sum(seeds)  # +['sf_goal']*2#['static','brown','cte_vel','sf_goal']
    horizon = 400
    evaluated_horizon = 400
    side_list = [25]

    folder_policy_list = []

    ###
    for i in range(len(policy_dir)):
        for SIDE in side_list:
            # Fixing save-dir mechanic to make it simpler
            savedir = '/home/alvaroserra/projects/gym_target_ig/ray_results'
            if policy_dir[i] != '':
                savedir = savedir + '/' + policy_dir[i]
            else:
                savedir = savedir + '/' + env
            # if not os.path.isdir(savedir):
            #    os.makedirs(savedir)
            experiment = experiment_list[i]
            expstr = '/exp' + str(experiment)
            video_dir = savedir + '/videos_exp_' + str(experiment) + '_ckpoint_' + str(ncheckpoint[i]) + '/'
            if heuristic_handcrafted[i]: video_dir = savedir + '/videos_exp_' + str(experiment) + '/'
            folder_name = video_dir + 'test_performance'
            if heuristic_policy[i]:
                folder_name = folder_name + '_heuristic'
            if heuristic_target_order[i]:
                folder_name = folder_name + '_heuristicTargetOrder'
            if horizon != 100:
                folder_name = folder_name + '_' + str(horizon)
            folder_name += '_histogram'
            if robot_robot_occlusion:
                folder_name = folder_name + '_robrobOcclusion'
            if robot_target_assignment[i]:
                folder_name = folder_name + '_rotarAssign'
            if SIDE != 8.0:
                folder_name = folder_name + '_' + str(SIDE)
            if realistic_Dengine[i]:
                folder_name = folder_name + '_realistic'
            if lh_ratio[i] != 1:
                folder_name = folder_name + '_' + str(lh_ratio[i])
            folder_name = folder_name + '_' + env_mode[i]
            # if i==0: folder_policy_list.append(folder_name + 'twice_observed.csv')
            folder_policy_list.append(folder_name + '.csv')

    stds = True
    #horizon *= 5
    ###
    #mode = 'success' # success OR nsteps
    extracted_data = dataextract(episodes, ntargets_list, folder_policy_list, relative_to_heuristic)
    hist_mean, hist_std, hist_ep_mean, hist_ep_std, hist_ep_tracked_mean, hist_ep_tracked_std,\
    hist_ep_nottracked_mean, hist_ep_nottracked_std, tracked_mean, tracked_std = extracthistograms(episodes, ntargets_list, folder_policy_list, relative_to_heuristic,seeds,horizon*lh_ratio[0])
    fig, ax = plt.subplots()
    for i, histdata in enumerate(hist_mean):
        plotlabel =  env_mode_name[i]
        ax.plot(np.arange(17),hist_mean[i],label=plotlabel)
        ax.fill_between(np.arange(17),
                    np.array(hist_mean[i]) - np.array(hist_std[i]),
                    np.array(hist_mean[i]) + np.array(hist_std[i]), alpha=0.5)
    ax.set_xlabel('# targets')
    ax.set_ylabel('Time (s)')
    ax.set_title('Distribution on target simultaneous observations')  # relative to Baseline 1')
    ax.legend(loc='upper right')
    #ax.set_xlim([0, 200])
    plt.savefig(video_dir + 'distribution.png')



    plot_x_axis = (np.arange(horizon*lh_ratio[0])+1)*0.05
    evaluated_x_axis = (evaluated_horizon*lh_ratio[0])*0.05


    fig, ax = plt.subplots()
    for i, histdata in enumerate(hist_mean):
        plotlabel = env_mode_name[i]
        ax.plot(plot_x_axis, hist_ep_mean[i], label=plotlabel)
        ax.fill_between(plot_x_axis,
                        np.array(hist_ep_mean[i]) - np.array(hist_ep_std[i]),
                        np.array(hist_ep_mean[i]) + np.array(hist_ep_std[i]), alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('# targets')
    ax.set_title('Simultaneous target observations along episode')  # relative to Baseline 1')
    ax.legend(loc='lower right')
    ax.set_xlim([0, evaluated_x_axis])
    plt.savefig(video_dir + 'target_ep.png')

    fig, ax = plt.subplots()
    for i, histdata in enumerate(hist_mean):
        plotlabel = env_mode_name[i]
        ax.plot(plot_x_axis, hist_ep_tracked_mean[i], label=plotlabel)
        ax.fill_between(plot_x_axis,
                        np.array(hist_ep_tracked_mean[i]) - np.array(hist_ep_tracked_std[i]),
                        np.array(hist_ep_tracked_mean[i]) + np.array(hist_ep_tracked_std[i]), alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('# targets')
    ax.set_title('Simultaneous tracked target observations along episode')  # relative to Baseline 1')
    ax.legend(loc='lower right')
    ax.set_xlim([0, evaluated_x_axis])
    plt.savefig(video_dir + 'tracked_target_ep.png')

    fig, ax = plt.subplots()
    for i, histdata in enumerate(hist_mean):
        plotlabel = env_mode_name[i]
        ax.plot(plot_x_axis, hist_ep_nottracked_mean[i], label=plotlabel)
        ax.fill_between(plot_x_axis,
                        np.array(hist_ep_nottracked_mean[i]) - np.array(hist_ep_nottracked_std[i]),
                        np.array(hist_ep_nottracked_mean[i]) + np.array(hist_ep_nottracked_std[i]), alpha=0.5)
    # ax2 = ax.twinx()
    # ax2.plot(np.arange(horizon) + 1, tracked_mean[-1],'g-.', label='% classified (Ours)')
    # ax2.fill_between(np.arange(horizon) + 1,
    #                 np.array(tracked_mean[-1]) - np.array(tracked_std[-1]),
    #                 np.array(tracked_mean[-1]) + np.array(tracked_std[-1]),color='g', alpha=0.5)
    # ax2.set_ylim([0,100])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('# targets')
    #ax2.set_ylabel('% classified (Ours)')

    ax.set_title('Simultaneous observations of unclassified targets')  # relative to Baseline 1')
    ax.legend(loc='upper right')
    ax.set_xlim([0, evaluated_x_axis])
    plt.savefig(video_dir + 'unclass_histogram_'+env_mode[0]+'.png')

    fig, ax = plt.subplots()
    for i, histdata in enumerate(hist_mean):
        plotlabel = env_mode_name[i]
        ax.plot(plot_x_axis, tracked_mean[i], label=plotlabel)
        ax.fill_between(plot_x_axis,
                     np.array(tracked_mean[i]) - np.array(tracked_std[i]),
                     np.array(tracked_mean[i]) + np.array(tracked_std[i]), alpha=0.5)
    # ax2 = ax.twinx()
    # ax2.plot(np.arange(horizon) + 1, tracked_mean[-1],'g-.', label='% classified (Ours)')
    # ax2.fill_between(np.arange(horizon) + 1,
    #                 np.array(tracked_mean[-1]) - np.array(tracked_std[-1]),
    #                 np.array(tracked_mean[-1]) + np.array(tracked_std[-1]),color='g', alpha=0.5)
    ax.set_ylim([0,100])
    ax.set_xlim([0,evaluated_x_axis])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('% classified')
    #ax2.set_ylabel('% classified (Ours)')

    ax.set_title('Percentage of classified targets')  # relative to Baseline 1')
    ax.legend(loc='upper left')
    plt.savefig(video_dir + 'class_percentage_' + env_mode[0] + '.png')

    plt.show()
"""
    ed_idx = 0
    seed_data_list = []
    for i in seeds:
        seed_data = compress_seed_data(extracted_data,ed_idx,i)
        seed_data_list.append(seed_data)
        ed_idx += i
    extracted_data = seed_data_list
    for mode in ['success','nsteps','ntracked', 'entropy']:
        fig, ax = plt.subplots()
        for i,datadict in enumerate(extracted_data):
            #plotlabel = 'heur. policy' if heuristic_policy[i] else ('heur. t. order' if heuristic_target_order[i] else 'learned policy')
            #plotlabel = plotlabel + ' + env ' + env_mode[i]
            plotlabel = env_mode_name[i]#'env: ' + env_mode_name[i]

            if mode == 'nsteps':
                ax.plot(datadict['ntargets'], datadict['episode_length_mean'],
                                  label=plotlabel)
                if stds:
                    ax.fill_between(np.array(datadict['ntargets']), np.array(datadict['episode_length_mean']) - np.array(datadict['episode_length_std']), np.array(datadict['episode_length_mean']) + np.array(datadict['episode_length_std']), alpha =0.5)
            if mode == 'success':
                ax.plot(datadict['ntargets'], datadict['success'],
                        label=plotlabel)
            if mode == 'ntracked':
                ax.plot(datadict['ntargets'], datadict['ntracked_mean'],
                                  label=plotlabel)
                if stds:
                    #ax.fill_between(np.array(datadict['ntargets']), np.array(datadict['ntracked_mean']) - np.array(datadict['ntracked_std']), np.array(datadict['ntracked_mean']) + np.array(datadict['ntracked_std']), alpha =0.5)
                    ax.fill_between(np.array(datadict['ntargets']),
                                    np.max([np.array(datadict['ntracked_mean']) - np.array(datadict['ntracked_std']),np.array(datadict['ntracked_min'])],axis=0),
                                np.min([np.array(datadict['ntracked_mean']) + np.array(datadict['ntracked_std']),np.array(datadict['ntracked_max'])],axis=0), alpha=0.5)
            if mode == 'entropy':
                ax.plot(datadict['ntargets'], datadict['entropy_mean'],
                        label=plotlabel)
                if stds:
                    # ax.fill_between(np.array(datadict['ntargets']), np.array(datadict['ntracked_mean']) - np.array(datadict['ntracked_std']), np.array(datadict['ntracked_mean']) + np.array(datadict['ntracked_std']), alpha =0.5)
                    ax.fill_between(np.array(datadict['ntargets']),
                                    np.max([np.array(datadict['entropy_mean']) - np.array(datadict['entropy_std']),np.array(datadict['entropy_min'])], axis=0),
                                    np.min([np.array(datadict['entropy_mean']) + np.array(datadict['entropy_std']),np.array(datadict['entropy_max'])], axis=0), alpha=0.5)
                    #ax.fill_between(np.array(datadict['ntargets']),np.array(datadict['ntracked_min']),np.array(datadict['ntracked_max']), alpha=0.5)

        ax.set_xlabel('number of targets')
        if mode == 'success':
            ax.set_ylabel('sucess rate')
            ax.set_title('Success rate')# relative to Baseline 1')
            ax.legend(loc='upper right')
            plt.savefig(video_dir+'success.png')
        elif mode == 'nsteps':
            ax.set_ylabel('number of steps')
            ax.set_title('Mean duration of Successful episodes')# relative to Baseline 1')
            ax.legend(loc='lower right')
            plt.savefig(video_dir+'nsteps.png')
        elif mode == 'ntracked':
            ax.set_ylabel('Percentage of targets tracked')
            ax.set_title('Targets tracked before Timeout')# relative to Baseline 1')
            ax.legend(loc='upper right')
            plt.savefig(video_dir+'ntracked.png')
        elif mode == 'entropy':
            ax.set_ylabel('Entropy')
            ax.set_title('Entropy before Timeout')# relative to Baseline 1')
            ax.legend(loc='upper left')
            plt.savefig(video_dir+'entropy.png')

    plt.show()
"""
