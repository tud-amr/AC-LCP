import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def dataextract(episodes, ntargets_list, foldernames, relative_to_heuristic):
    info_model = []
    bdataframe = pd.read_csv(foldernames[0], header= None)
    for i in range(0,len(foldernames)):
        dataframe = pd.read_csv(foldernames[i], header= None)
        auxDict = {'ntargets':[],'episode_length_mean':[],'episode_length_std':[],'success':[],'ntracked_mean':[],'ntracked_std':[],'ntracked_min':[],'ntracked_max':[],
                   'entropy_mean':[],'entropy_std':[],'entropy_min':[],'entropy_max':[],'reward_mean':[],'reward_std':[],'reward_min':[],'reward_max':[]} #'success_std':[]}
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
            relRewardData = dataframe[dataframe[2] == numtarget][7]
            if relative_to_heuristic:
                relRewardData -= bdataframe[dataframe[2] == numtarget][7]
            auxDict['reward_mean'].append(relRewardData.mean())
            auxDict['reward_std'].append(relRewardData.std())
            auxDict['reward_min'].append(relRewardData.min())
            auxDict['reward_max'].append(relRewardData.max())

        info_model.append(auxDict)
    return info_model

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
    #print(rolloutArgs.checkpoint)
    #### in code trials
    #args.run = 'MixinPPO'
    episodes = 25
    nrobots_list = [1,2]
    ntargets_list = [1,3,5,7,9,11,13,15,17,19,21,23,25]#,55]
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
    policy_dir = ["heuristic_policy_config", "heuristic_policy_config"]
    ncheckpoint = [1,
                   1]#[3000, 6000, 10710]#[800,800]#[1090, 1090]#[950,950,1050,450,400] #[950,950,950,950] #460
    experiment_list = [1,1]#[19, 4, 1] #[1,1,2,1,1]#[1,1,1,1] #3#3  # 1

    n_testmodels = 0
    heuristic_handcrafted = [True,
                             True]
    heuristic_policy = [True,
                        True]
    heuristic_target_order = [True,
                              True,]
    seeds = [1,1]
    #static_env = [True, True,True]#[True,True, True,True,True] #[False, False, True, True]
    env_mode = ['cte_vel']*(2) #+['sf_goal']*2#['static','brown','cte_vel','sf_goal']
    #env_mode_name = ['baseline 1','baseline 2 seed 1', 'baseline 2 seed 2','baseline 2 seed 3','seed 1','seed 2','seed 3']#+['baseline 2', 'baseline 3 (wip)']
    env_mode_name = ['baseline 1 - twice Observed', 'baseline 1 - 2 robots']  # +['baseline 2', 'baseline 3 (wip)']
    horizon = 100

    folder_policy_list = []

    ###
    for i in range(len(policy_dir)):
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
        folder_name = folder_name + '_' + env_mode[i]
        if i==0: folder_policy_list.append(folder_name + 'twice_observed.csv')
        else: folder_policy_list.append(folder_name + '.csv')


    stds = True
    ###
    #mode = 'success' # success OR nsteps
    extracted_data = dataextract(episodes, ntargets_list, folder_policy_list, relative_to_heuristic)

    ed_idx = 0
    seed_data_list = []
    for i in seeds:
        seed_data = compress_seed_data(extracted_data,ed_idx,i)
        seed_data_list.append(seed_data)
        ed_idx += i
    #extracted_data = seed_data_list
    for mode in ['success','nsteps','ntracked', 'entropy', 'rewards']:
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
            ax.legend(loc='upper right')
            plt.savefig(video_dir+'ntracked.png')
        elif mode == 'entropy':
            ax.set_ylabel('Entropy')
            ax.set_title('Entropy before Timeout')# relative to Baseline 1')
            ax.legend(loc='upper left')
            plt.savefig(video_dir+'entropy.png')
        elif mode == 'rewards':
            ax.set_ylabel('Rewards')
            ax.set_title('Rewards before Timeout')# relative to Baseline 1')
            ax.legend(loc='upper left')
            plt.savefig(video_dir+'reward.png')

    plt.show()

