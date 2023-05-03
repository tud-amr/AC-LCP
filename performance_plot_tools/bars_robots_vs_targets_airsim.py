import os
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def data_colormap(data_list, mode_list, bounds, title):
    """
    Helper function to plot data with associated colormap.
    """
    viridis = cm.get_cmap('viridis', 101)
    pink = np.array([248/256, 24/256, 148/256, 1])
    newcolors = viridis(np.linspace(0,1,101))
    newcolors[-1] = pink
    newcmp = ListedColormap(newcolors)
    colormaps = [newcmp]
    x_values = [1, 2, 3, 4]
    xticks = [1, 2, 3, 4]
    y_values = [10, 20, 30, 40]
    yticks = [1, 2, 3, 4]
    #np.random.seed(19680801)
    #data = np.random.randn(30, 30)*10
    #n = len(colormaps)
    n = len(data_list)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    cmap = colormaps[0]
    i=0
    for [ax, data] in zip(axs.flat, data_list):
        data[np.isnan(data)] = 500
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=bounds[0], vmax=bounds[1])
        fig.colorbar(psm, ax=ax)
        ax.set_xlabel('number of robots')
        ax.set_ylabel('number of targets')
        ax.set_xticks(xticks,x_values)
        ax.set_yticks(yticks,y_values)
        ax.set_title(mode_list[i])
        i+=1
    #plt.show()
    fig.suptitle(title)
    return fig, axs


def dataextract(episodes, ntargets_list, foldernames, relative_to_heuristic, nrobots_list):
    info_model = []
    bdataframe = pd.read_csv(foldernames[0], header= None)
    for i in range(0,len(foldernames)):
        dataframe = pd.read_csv(foldernames[i], header= None)
        auxDict = {'ntargets':[],'nrobots':[], 'episode_length_mean':[],'episode_length_std':[],'success':[],'ntracked_mean':[],'ntracked_std':[],'ntracked_min':[],'ntracked_max':[],
                   'entropy_mean':[],'entropy_std':[],'entropy_min':[],'entropy_max':[],'reward_mean':[],'reward_std':[],'reward_min':[],'reward_max':[]} #'success_std':[]}
        for numtarget in ntargets_list:
            for numrobot in nrobots_list:
                auxDict['ntargets'].append(numtarget)
                auxDict['nrobots'].append(numrobot)
                #relSuccessData = dataframe[dataframe[2]==numtarget][4]
                relSuccessData = dataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][5]==numtarget
                if relative_to_heuristic:
                    relSuccessData -= bdataframe[dataframe[2]==numtarget][dataframe[1]==numrobot][4]
                auxDict['success'].append(relSuccessData.mean())
                #relStepData = dataframe[dataframe[2] == numtarget][dataframe[4]==1][3]
                relStepData = dataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][dataframe[5] == numtarget][3]
                if relative_to_heuristic:
                    relStepData -= bdataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][dataframe[4]==1][3]
                auxDict['episode_length_mean'].append(relStepData.mean())
                auxDict['episode_length_std'].append(relStepData.std())
                relTrackedData = dataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][5]
                if relative_to_heuristic:
                    relTrackedData -= bdataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][5]
                #relTrackedData = relTrackedData / numtarget *100
                relTrackedData = relTrackedData / numrobot
                auxDict['ntracked_mean'].append(relTrackedData.mean())
                auxDict['ntracked_std'].append(relTrackedData.std())
                auxDict['ntracked_min'].append(relTrackedData.min())
                auxDict['ntracked_max'].append(relTrackedData.max())
                relEntropyData = dataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][6]
                if relative_to_heuristic:
                    relEntropyData -= bdataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][6]
                auxDict['entropy_mean'].append(relEntropyData.mean())
                auxDict['entropy_std'].append(relEntropyData.std())
                auxDict['entropy_min'].append(relEntropyData.min())
                auxDict['entropy_max'].append(relEntropyData.max())
                relRewardData = dataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][7]
                if relative_to_heuristic:
                    relRewardData -= bdataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][7]
                auxDict['reward_mean'].append(relRewardData.mean()/numrobot)
                auxDict['reward_std'].append(relRewardData.std()/numrobot)
                auxDict['reward_min'].append(relRewardData.min()/numrobot)
                auxDict['reward_max'].append(relRewardData.max()/numrobot)

        info_model.append(auxDict)
    return info_model

def compress_seed_data(extracted_data,ed_idx,nseeds):
    important_data = extracted_data[ed_idx:ed_idx+nseeds]
    auxDict = {'ntargets': [],'nrobots':[], 'episode_length_mean': [], 'episode_length_std': [], 'success': [], 'ntracked_mean': [],
               'ntracked_std': [], 'ntracked_min': [], 'ntracked_max': [],
               'entropy_mean': [], 'entropy_std': [], 'entropy_min': [], 'entropy_max': [],
               'reward_mean': [], 'reward_std': [], 'reward_min': [], 'reward_max': [],}  # 'success_std':[]}
    for key in auxDict:
        for i in range(nseeds):
            if key=='ntargets' and i>0: break
            auxDict[key].append(important_data[i][key])
        if key!='ntargets':
            auxDict[key] = np.array(auxDict[key])

    auxDict['ntargets'] = auxDict['ntargets'][0]
    auxDict['nrobots'] = auxDict['nrobots'][0]
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
    auxvar = auxDict['reward_mean'].copy()
    auxDict['reward_mean'] = list(auxvar.mean(axis=0))
    auxDict['reward_std'] = list(auxvar.std(axis=0))
    auxDict['reward_min'] = list(auxvar.min(axis=0))
    auxDict['reward_max'] = list(auxvar.max(axis=0))

    return auxDict





if __name__ == '__main__':
    #print(rolloutArgs.checkpoint)
    #### in code trials
    #args.run = 'MixinPPO'
    episodes = 50
    nrobots_list = [1,2,3,4]
    ntargets_list = [10,20,30,40]#,55]
    relative_to_heuristic = False
    ####
    # policy_dir = 'fully_connected_max10targets_longtrain'
    # policy_dir = 'transformer_stable_nodropout_largetrain'
    # policy_dir = 'deepsets_largetrain_1to3agents'
    # policy_dir = 'deepsets_largetrain_1to3agents_restored'

    # policy_dir = ["heuristic_policy_config","baseline_2_3_upto_3_targets"]+["policy_versions/attentionVsDeepSets/cte_vel/deep_sets",
    #                                                                         "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/seed100/1_12_targets/low_ntargets",
    #                                                                         "policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/seed200/1_12_targets/low_ntargets"]
    # policy_dir = ["transf_st_drop_1layer_1target","transf_st_drop_1layer_1target", "transf_st_drop_1layer_1target",
    # "transf_st_drop_1layer_1target_static_6t","transf_st_drop_1layer_1target_static_8t"]#["transf_st_drop_1layer_1target",
    # "transf_st_drop_1layer_1target","transf_st_drop_1layer_1target","transf_st_drop_1layer_1target"]
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
        # "CORL2022/our_method_airper/seed100/1stphase",
        # "Ours/setTransformer_opt_v2/50x50_env/seed200/2ndphase",
        # "Ours/setTransformer_opt_v2/50x50_env/seed300/2ndphase",
        # "Ours/setTransformer_opt_v2/50x50_env/seed400/2ndphase",
        # "Ours/setTransformer_opt_v2/50x50_env/seed500/2ndphase"
    ]
    policy_dir += addendum
    ncheckpoint += [6000] * len(addendum)
    experiment_list += [1] * len(addendum)
    heuristic_handcrafted += [False] * len(addendum)
    heuristic_policy += [False] * len(addendum)
    heuristic_target_order += [False] * len(addendum)
    robot_target_assignment += [False] * len(addendum)
    realistic_Dengine += [False] * len(addendum)
    lh_ratio += [1] * len(addendum)
    seeds += [len(addendum)] * int(len(addendum) != 0)
    env_mode_name += ['Multi-target (Ours)'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)

    # RSS paper
    addendum = [
        "CORL2022/our_method_airper/seed100/2ndphase_v2",
        # "Ours/setTransformer_opt_v2/50x50_env/seed200/2ndphase",
        # "Ours/setTransformer_opt_v2/50x50_env/seed300/2ndphase",
        # "Ours/setTransformer_opt_v2/50x50_env/seed400/2ndphase",
        # "Ours/setTransformer_opt_v2/50x50_env/seed500/2ndphase"
    ]
    policy_dir += addendum
    ncheckpoint += [5750] * len(addendum)
    experiment_list += [1] * len(addendum)
    heuristic_handcrafted += [False] * len(addendum)
    heuristic_policy += [False] * len(addendum)
    heuristic_target_order += [False] * len(addendum)
    robot_target_assignment += [False] * len(addendum)
    realistic_Dengine += ["airsim"] * len(addendum)
    lh_ratio += [1] * len(addendum)
    seeds += [len(addendum)] * int(len(addendum) != 0)
    env_mode_name += ['Multi-target (Ours)'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)

    # LSTM baseline
    addendum = [
        # "LSTM/seed100/2ndphase",
        # "LSTM/seed200/2ndphase",
        # "LSTM/seed300/2ndphase",
        # "LSTM/seed400/2ndphase",
        # "LSTM/seed500/2ndphase"
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
        "sim_per/baseline2_v2/seed100",
        # "baseline_2/seed200",
        # "baseline_2/seed300",
        # "baseline_2/seed400",
        # "baseline_2/seed500",
    ]
    policy_dir += addendum
    ncheckpoint += [6000] * len(addendum)
    experiment_list += [1]
    heuristic_handcrafted += [False] * len(addendum)
    heuristic_policy += [True] * len(addendum)
    heuristic_target_order += [True] * len(addendum)
    robot_target_assignment += [True] * len(addendum)
    realistic_Dengine += ["airsim"] * len(addendum)
    lh_ratio += [1] * len(addendum)
    seeds += [len(addendum)] * int(len(addendum) != 0)
    env_mode_name += ['Single-target (Ours)'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)

    # Hard-coded baseline
    addendum = [
        "CORL2022/baseline_1",
    ]
    policy_dir += addendum
    ncheckpoint += [1] * len(addendum)
    experiment_list += [1] * len(addendum)
    heuristic_handcrafted += [True] * len(addendum)
    heuristic_policy += [True] * len(addendum)
    heuristic_target_order += [True] * len(addendum)
    robot_target_assignment += [True] * len(addendum)
    realistic_Dengine += ["airsim"] * len(addendum)
    lh_ratio += [1] * len(addendum)
    seeds += [len(addendum)] * int(len(addendum) != 0)
    env_mode_name += ['Hand-crafted'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)

    env_mode = ['airsim'] * np.sum(seeds)  # +['sf_goal']*2#['static','brown','cte_vel','sf_goal']
    horizon = 400
    evaluated_horizon = 300
    side_list = [25]

    number_of_classes = 2
    dyn_sigma = 0
    simulated_perception = "airsim"

    folder_policy_list = []

    ###
    for i in range(len(policy_dir)):
        for SIDE in side_list:
            # Fixing save-dir mechanic to make it simpler
            savedir = '/home/amr/projects/alvaro/gym_target_ig/ray_results'
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
            if realistic_Dengine[i] == "dummy_env":
                folder_name = folder_name + '_realistic'
            elif realistic_Dengine[i] == "airsim":
                folder_name = folder_name + '_airsim'
            if lh_ratio[i] != 1:
                folder_name = folder_name + '_' + str(lh_ratio[i])
            if number_of_classes != 2:
                folder_name = folder_name + '_' + str(number_of_classes) + 'classes'
            if dyn_sigma != 0:
                folder_name = folder_name + '_dynSigma' + str(dyn_sigma)
            if simulated_perception != "dummy":
                folder_name = folder_name + '_per_' + str(simulated_perception)
            folder_name = folder_name + '_slowmo'

            folder_name = folder_name + '_' + env_mode[i]
            # if i==0: folder_policy_list.append(folder_name + 'twice_observed.csv')
            # folder_policy_list.append(folder_name + '_teleport.csv')
            if i==0: folder_policy_list.append(folder_name + '_dronesoverlap.csv')
            else: folder_policy_list.append(folder_name + '.csv')

    stds = False
    ###
    #mode = 'success' # success OR nsteps
    extracted_data = dataextract(episodes, ntargets_list, folder_policy_list, relative_to_heuristic, nrobots_list)

    ed_idx = 0
    seed_data_list = []
    for i in seeds:
        seed_data = compress_seed_data(extracted_data,ed_idx,i)
        seed_data_list.append(seed_data)
        ed_idx += i
    extracted_data = seed_data_list
    for mode in ['success','nsteps','ntracked', 'entropy', 'rewards']:
        fig, ax = plt.subplots()
        """
        if mode == 'nsteps':
            fig, ax = data_colormap(
                [np.array(extracted_data[rrr]['episode_length_mean']).reshape(len(ntargets_list), len(nrobots_list)) for
                 rrr in range(len(extracted_data))],
                env_mode_name,
                [0,500], "Mean duration of Successful episodes")

        if mode == 'success':
            fig, ax = data_colormap(
                [np.array(extracted_data[rrr]['success']).reshape(len(ntargets_list), len(nrobots_list)) for
                 rrr in range(len(extracted_data))],
                env_mode_name,
                [0,1], "Success rate")

        if mode == 'ntracked':
            fig, ax = data_colormap(
                [np.array(extracted_data[rrr]['ntracked_mean']).reshape(len(ntargets_list), len(nrobots_list)) for
                 rrr in range(len(extracted_data))],
                env_mode_name,
                [0,20], "Targets tracked per robot before Timeout")


        if mode == 'entropy':
            fig, ax = data_colormap(
                [np.array(extracted_data[rrr]['entropy_mean']).reshape(len(ntargets_list), len(nrobots_list)) for
                 rrr in range(len(extracted_data))],
                env_mode_name,
                [0,20], "Entropy before Timeout")

        if mode == 'rewards':
            fig, ax = data_colormap(
                [np.array(extracted_data[rrr]['reward_mean']).reshape(len(ntargets_list), len(nrobots_list)) for
                 rrr in range(len(extracted_data))],
                env_mode_name,
                #[-100,250],
                [-25, 50],
                "Reward per robot before Timeout")
        """
        for i,datadict in enumerate(extracted_data):
            if plot_curve[i]:
                #plotlabel = 'heur. policy' if heuristic_policy[i] else ('heur. t. order' if heuristic_target_order[i] else 'learned policy')
                #plotlabel = plotlabel + ' + env ' + env_mode[i]
                plotlabel = env_mode_name[i]#'env: ' + env_mode_name[i]


                if mode == 'success':
                    data=[np.array(extracted_data[rrr]['success']).reshape(len(ntargets_list), len(nrobots_list)) for
                     rrr in range(len(extracted_data))]
                    scale = [2,3,4]
                    ax.plot(scale, [data[i][aux][aux] for aux in range(len(scale))],
                            label=plotlabel)

                if mode == 'ntracked':
                    data = [
                        np.array(extracted_data[rrr]['ntracked_mean']).reshape(len(ntargets_list), len(nrobots_list))
                        for
                        rrr in range(len(extracted_data))]
                    datamin = [
                        np.array(extracted_data[rrr]['ntracked_min']).reshape(len(ntargets_list), len(nrobots_list))
                        for
                        rrr in range(len(extracted_data))]
                    datamax = [
                        np.array(extracted_data[rrr]['ntracked_max']).reshape(len(ntargets_list), len(nrobots_list))
                        for
                        rrr in range(len(extracted_data))]
                    datastd = [
                        np.array(extracted_data[rrr]['ntracked_std']).reshape(len(ntargets_list), len(nrobots_list))
                        for
                        rrr in range(len(extracted_data))]
                    scale = [1, 2, 3, 4]
                    ax.plot(scale, [data[i][aux][aux] for aux in range(len(scale))],
                            label=plotlabel)
                    if stds:
                        # ax.fill_between(np.array(datadict['ntargets']), np.array(datadict['ntracked_mean']) - np.array(datadict['ntracked_std']), np.array(datadict['ntracked_mean']) + np.array(datadict['ntracked_std']), alpha =0.5)
                        ax.fill_between(np.array(datadict['nrobots']),
                                        np.max(
                                            [np.array(datadict['ntracked_mean']) - np.array(datadict['ntracked_std']),
                                             np.array(datadict['ntracked_min'])], axis=0),
                                        np.min(
                                            [np.array(datadict['ntracked_mean']) + np.array(datadict['ntracked_std']),
                                             np.array(datadict['ntracked_max'])], axis=0), alpha=0.5)
                if mode == 'nsteps':
                    data = [
                        np.array(extracted_data[rrr]['episode_length_mean']).reshape(len(ntargets_list), len(nrobots_list))
                        for
                        rrr in range(len(extracted_data))]
                    scale = [1, 2, 3, 4]
                    ax.plot(scale, [data[i][aux][aux] for aux in range(len(scale))],
                            label=plotlabel)
                    if stds:
                        # ax.fill_between(np.array(datadict['ntargets']), np.array(datadict['ntracked_mean']) - np.array(datadict['ntracked_std']), np.array(datadict['ntracked_mean']) + np.array(datadict['ntracked_std']), alpha =0.5)
                        ax.fill_between(np.array(datadict['nrobots']),
                                        np.max(
                                            [np.array(datadict['episode_length_mean']) - np.array(datadict['episode_length_std']),
                                             np.array(datadict['episode_length_min'])], axis=0),
                                        np.min(
                                            [np.array(datadict['episode_length_mean']) + np.array(datadict['episode_length_std']),
                                             np.array(datadict['episode_length_max'])], axis=0), alpha=0.5)

                if mode == 'entropy':
                    data = [np.array(extracted_data[rrr]['entropy_mean']).reshape(len(ntargets_list), len(nrobots_list)) for
                            rrr in range(len(extracted_data))]
                    scale = [1, 2, 3, 4]
                    ax.plot(scale, [data[i][aux][aux] for aux in range(len(scale))],
                            label=plotlabel)
                    if stds:
                        # ax.fill_between(np.array(datadict['ntargets']), np.array(datadict['ntracked_mean']) - np.array(datadict['ntracked_std']), np.array(datadict['ntracked_mean']) + np.array(datadict['ntracked_std']), alpha =0.5)
                        ax.fill_between(np.array(datadict['nrobots']),
                                        np.max([np.array(datadict['entropy_mean']) - np.array(datadict['entropy_std']),np.array(datadict['entropy_min'])], axis=0),
                                        np.min([np.array(datadict['entropy_mean']) + np.array(datadict['entropy_std']),np.array(datadict['entropy_max'])], axis=0), alpha=0.5)

                if mode == 'rewards':
                    data = [np.array(extracted_data[rrr]['reward_mean']).reshape(len(ntargets_list), len(nrobots_list)) for
                            rrr in range(len(extracted_data))]
                    scale = [1, 2, 3, 4]
                    ax.plot(scale, [data[i][aux][aux] for aux in range(len(scale))],
                            label=plotlabel)
                    if stds:
                        # ax.fill_between(np.array(datadict['ntargets']), np.array(datadict['ntracked_mean']) - np.array(datadict['ntracked_std']), np.array(datadict['ntracked_mean']) + np.array(datadict['ntracked_std']), alpha =0.5)
                        ax.fill_between(np.array(datadict['nrobots']),
                                        np.max(
                                            [np.array(datadict['reward_mean']) - np.array(datadict['reward_std']),
                                             np.array(datadict['reward_min'])], axis=0),
                                        np.min(
                                            [np.array(datadict['reward_mean']) + np.array(datadict['reward_std']),
                                             np.array(datadict['reward_max'])], axis=0), alpha=0.5)

                        #ax.fill_between(np.array(datadict['ntargets']),np.array(datadict['ntracked_min']),np.array(datadict['ntracked_max']), alpha=0.5)

            plt.xlabel('Problem scale (x 10 targets/robot)')
            if mode == 'success':
                plt.ylabel('success rate')
                plt.title('Success rate')# relative to Baseline 1')
                plt.legend(loc='lower right')
                plt.savefig(video_dir+'success.png')
            elif mode == 'nsteps':
                plt.ylabel('number of steps')
                plt.title('Mean duration of Successful episodes')# relative to Baseline 1')
                plt.legend(loc='upper right')
                plt.savefig(video_dir+'nsteps.png')
            elif mode == 'ntracked':
                plt.ylabel('Percentage of targets tracked')
                plt.title('Targets tracked before Timeout')# relative to Baseline 1')
                plt.legend(loc='lower right')
                plt.savefig(video_dir+'ntracked.png')
            elif mode == 'entropy':
                plt.ylabel('Entropy')
                plt.title('Entropy before Timeout')# relative to Baseline 1')
                plt.legend(loc='lower left')
                plt.savefig(video_dir+'entropy.png')
            elif mode == 'rewards':
                plt.ylabel('Rewards')
                plt.title('Reward before Timeout')# relative to Baseline 1')
                plt.legend(loc='lower right')
                plt.savefig(video_dir+'reward.png')

    plt.show()

