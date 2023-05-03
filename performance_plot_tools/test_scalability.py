import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def dataextract(episodes, ntargets_list, foldernames, relative_to_heuristic, nrobots_list, side_list,lh_ratio):
    info_model = []
    bdataframe = pd.read_csv(foldernames[0], header= None)
    for i in range(0,len(foldernames)):
        dataframe = pd.read_csv(foldernames[i], header= None)
        if i % len(side_list) == 0:
            auxDict = {'ntargets':[],'nrobots':[],'side':[], 'episode_length_mean':[],'episode_length_std':[],'success':[],'ntracked_mean':[],'ntracked_std':[],'ntracked_min':[],'ntracked_max':[],
                       'entropy_mean':[],'entropy_std':[],'entropy_min':[],'entropy_max':[],'reward_mean':[],'reward_std':[],'reward_min':[],'reward_max':[]} #'success_std':[]}
        for numtarget in ntargets_list:
            for numrobot in nrobots_list:
                auxDict['ntargets'].append(numtarget)
                auxDict['nrobots'].append(numrobot)
                auxDict['side'].append(side_list[i%len(side_list)]*2)
                #relSuccessData = dataframe[dataframe[2]==numtarget][4]
                relSuccessData = dataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][5]==numtarget
                if relative_to_heuristic:
                    relSuccessData -= bdataframe[dataframe[2]==numtarget][dataframe[1]==numrobot][4]
                auxDict['success'].append(relSuccessData.mean())
                #relStepData = dataframe[dataframe[2] == numtarget][dataframe[4]==1][3]
                relStepData = dataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][dataframe[5] == numtarget][3] / lh_ratio[i]
                if relative_to_heuristic:
                    relStepData -= bdataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][dataframe[4]==1][3] / lh_ratio[i]
                auxDict['episode_length_mean'].append(relStepData.mean())
                auxDict['episode_length_std'].append(relStepData.std())
                relTrackedData = dataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][5]
                if relative_to_heuristic:
                    relTrackedData -= bdataframe[dataframe[2] == numtarget][dataframe[1]==numrobot][5]
                relTrackedData = relTrackedData / numtarget *100
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
        if i % len(side_list) == 0:
            info_model.append(auxDict)
    return info_model

def compress_seed_data(extracted_data,ed_idx,nseeds):
    important_data = extracted_data[ed_idx:ed_idx+nseeds]
    auxDict = {'ntargets': [],'nrobots':[],'side':[], 'episode_length_mean': [], 'episode_length_std': [], 'success': [], 'ntracked_mean': [],
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
    auxDict['side'] = auxDict['side'][0]
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
    episodes = 50
    nrobots_list = [1]
    ntargets_list = [1,10,20,30,40]  # ,55]
    # env = 'SceneEnv_v3'
    relative_to_heuristic = False

    policy_dir = []
    ncheckpoint = []
    experiment_list = []

    n_testmodels = 0
    heuristic_handcrafted = []
    heuristic_policy = []
    heuristic_target_order = []
    robot_robot_occlusion = True
    robot_target_assignment = []
    realistic_Dengine = []

    lh_ratio = []

    seeds = []
    env_mode_name = []

    plot_curve = []
    number_of_classes = []

    # RSS paper
    addendum = [
                # "CORL2022/Ours/setTransformer_opt_v2/50x50_env/seed100/2ndphase",
                "CORL2022/Ours/setTransformer_opt_v2/50x50_env/seed200/2ndphase",
                # "CORL2022/Ours/setTransformer_opt_v2/50x50_env/seed300/2ndphase",
                # "CORL2022/Ours/setTransformer_opt_v2/50x50_env/seed400/2ndphase",
                # "CORL2022/Ours/setTransformer_opt_v2/50x50_env/seed500/2ndphase",
                ]
    policy_dir += addendum
    ncheckpoint += [8000]*len(addendum)
    experiment_list += [1]*len(addendum)
    heuristic_handcrafted += [False]*len(addendum)
    heuristic_policy += [False]*len(addendum)
    heuristic_target_order += [False]*len(addendum)
    robot_target_assignment += [False]*len(addendum)
    realistic_Dengine += [True]*len(addendum)
    lh_ratio += [5]*len(addendum)
    seeds += [len(addendum)]*int(len(addendum)!=0)
    env_mode_name += ['Ours']*int(len(addendum)!=0)
    plot_curve += [True]*int(len(addendum)!=0)
    number_of_classes += [2]*len(addendum)


    # RSS paper
    addendum = [
        # "CORL2022/Ours/setTransformer_opt_v2/50x50_env/seed100/2ndphase",
        # "CORL2022/Ours/setTransformer_opt_v2/50x50_env/seed200/2ndphase",
        # "CORL2022/Ours/setTransformer_opt_v2/50x50_env/seed300/2ndphase",
        # "CORL2022/Ours/setTransformer_opt_v2/50x50_env/seed400/2ndphase",
        # "CORL2022/Ours/setTransformer_opt_v2/50x50_env/seed500/2ndphase",
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
    env_mode_name += ['Ours - 10 classes'] * int(len(addendum) != 0)
    plot_curve += [False] * int(len(addendum) != 0)
    number_of_classes += [10] * len(addendum)

    # Deepsets paper
    addendum = [
        # "CORL2022/DeepSets/seed100/2ndphase",
        # "CORL2022/DeepSets/seed200/2ndphase",
        # "CORL2022/DeepSets/seed300/2ndphase",
        # "CORL2022/DeepSets/seed400/2ndphase",
        # "CORL2022/DeepSets/seed500/2ndphase",
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
    env_mode_name += ['Attention + DeepSets [22]'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)
    number_of_classes += [2] * len(addendum)


    # # LSTM baseline
    addendum = [
        "CORL2022/LSTM/seed100/2ndphase",
        # "CORL2022/LSTM/seed200/2ndphase",
        # "CORL2022/LSTM/seed300/2ndphase",
        # "CORL2022/LSTM/seed400/2ndphase",
        # "CORL2022/LSTM/seed500/2ndphase"
    ]
    policy_dir += addendum
    ncheckpoint += [8000]*len(addendum)
    experiment_list += [1]*len(addendum)
    heuristic_handcrafted += [False]*len(addendum)
    heuristic_policy += [False]*len(addendum)
    heuristic_target_order += [True]*len(addendum)
    robot_target_assignment += [True]*len(addendum)
    realistic_Dengine += [True]*len(addendum)
    lh_ratio += [5] * len(addendum)
    seeds += [len(addendum)] * int(len(addendum) != 0)
    env_mode_name += ['LSTM encoder'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)
    number_of_classes += [2] * len(addendum)

    # Single-target baseline
    addendum = [
        # "CORL2022/baseline_2/seed100",
        # "CORL2020/baseline_2/seed200",
        # "CORL2020/baseline_2/seed300",
        # "CORL2020/baseline_2/seed400",
        # "CORL2020/baseline_2/seed500",
    ]
    policy_dir += addendum
    ncheckpoint += [6000] * len(addendum)
    experiment_list += [1] * len(addendum)
    heuristic_handcrafted += [False] * len(addendum)
    heuristic_policy += [True] * len(addendum)
    heuristic_target_order += [True] * len(addendum)
    robot_target_assignment += [True] * len(addendum)
    realistic_Dengine += [False] * len(addendum)
    lh_ratio += [1] * len(addendum)
    seeds += [len(addendum)] * int(len(addendum) != 0)
    env_mode_name += ['Single-target'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)
    number_of_classes += [2] * int(len(addendum) != 0)

    # Hard-coded baseline
    addendum = [
        # "CORL2022/baseline_1",
    ]
    policy_dir += addendum
    ncheckpoint += [6000] * len(addendum)
    experiment_list += [1] * len(addendum)
    heuristic_handcrafted += [True] * len(addendum)
    heuristic_policy += [True] * len(addendum)
    heuristic_target_order += [True] * len(addendum)
    robot_target_assignment += [True] * len(addendum)
    realistic_Dengine += [False] * len(addendum)
    lh_ratio += [1] * len(addendum)
    seeds += [len(addendum)] * int(len(addendum) != 0)
    env_mode_name += ['Hand-crafted'] * int(len(addendum) != 0)
    plot_curve += [True] * int(len(addendum) != 0)
    number_of_classes += [2] * int(len(addendum) != 0)


    env_mode = ['sf_goal'] * np.sum(seeds)  # +['sf_goal']*2#['static','brown','cte_vel','sf_goal']
    horizon = 400
    side_list = [25]


    folder_policy_list = []

    ###
    for i in range(len(policy_dir)):
        for SIDE in side_list:
            # Fixing save-dir mechanic to make it simpler
            savedir = '/home/amr/projects/gym_target_ig/ray_results'
            if policy_dir[i] != '':
                savedir = savedir + '/' + policy_dir[i]
            else:
                savedir = savedir + '/' + env
            # if not os.path.isdir(savedir):
            #    os.makedirs(savedir)
            experiment = experiment_list[i]
            expstr = '/exp' + str(experiment)
            video_dir = savedir + '/videos_exp_' + str(experiment) + '_ckpoint_' + str(ncheckpoint[i])+'/'
            if heuristic_handcrafted[i]: video_dir = savedir + '/videos_exp_'+str(experiment)+'/'
            folder_name = video_dir + 'test_performance'
            if heuristic_policy[i]:
                folder_name = folder_name + '_heuristic'
            if heuristic_target_order[i]:
                folder_name = folder_name + '_heuristicTargetOrder'
            if horizon!=100:
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
            if number_of_classes[i] != 2:
                folder_name = folder_name + '_' + str(number_of_classes[i]) + 'classes'
            folder_name = folder_name + '_' + env_mode[i]
            # if i==0: folder_policy_list.append(folder_name + 'twice_observed.csv')
            folder_policy_list.append(folder_name + '.csv')


    stds = True
    ###
    #mode = 'success' # success OR nsteps
    extracted_data = dataextract(episodes, ntargets_list, folder_policy_list, relative_to_heuristic, nrobots_list, side_list, lh_ratio)

    ed_idx = 0
    seed_data_list = []
    for i in seeds:
        seed_data = compress_seed_data(extracted_data,ed_idx,i)
        seed_data_list.append(seed_data)
        ed_idx += i
    extracted_data = seed_data_list
    for mode in ['success','nsteps','ntracked', 'entropy', 'rewards']:
        fig, ax = plt.subplots()
        for i,datadict in enumerate(extracted_data):
            if plot_curve[i]:
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

                if mode == 'rewards':
                    ax.plot(datadict['ntargets'], datadict['reward_mean'],
                            label=plotlabel)
                    if stds:
                        # ax.fill_between(np.array(datadict['ntargets']), np.array(datadict['ntracked_mean']) - np.array(datadict['ntracked_std']), np.array(datadict['ntracked_mean']) + np.array(datadict['ntracked_std']), alpha =0.5)
                        ax.fill_between(np.array(datadict['ntargets']),
                                        np.max(
                                            [np.array(datadict['reward_mean']) - np.array(datadict['reward_std']),
                                             np.array(datadict['reward_min'])], axis=0),
                                        np.min(
                                            [np.array(datadict['reward_mean']) + np.array(datadict['reward_std']),
                                             np.array(datadict['reward_max'])], axis=0), alpha=0.5)

                        #ax.fill_between(np.array(datadict['ntargets']),np.array(datadict['ntracked_min']),np.array(datadict['ntracked_max']), alpha=0.5)

            ax.set_xlabel('number of targets')
            if mode == 'success':
                ax.set_ylabel('success rate')
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
                ax.legend(loc='lower left')
                plt.savefig(video_dir+'ntracked.png')
            elif mode == 'entropy':
                ax.set_ylabel('Entropy')
                ax.set_title('Entropy before Timeout')# relative to Baseline 1')
                ax.legend(loc='upper left')
                plt.savefig(video_dir+'entropy.png')
            elif mode == 'rewards':
                ax.set_ylabel('Rewards')
                ax.set_title('Reward before Timeout')# relative to Baseline 1')
                ax.legend(loc='lower right')
                plt.savefig(video_dir+'reward.png')

    plt.show()

