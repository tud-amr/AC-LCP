import ray
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune import run
from ray.tune.registry import register_env

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf, try_import_tfp

from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.agents.ppo.ppo_tf_policy import ValueNetworkMixin, KLCoeffMixin

from ray.rllib.utils.tf_ops import make_tf_callable

# parser
from utils.parse_args import parse_args
import time

# callbacks

# configs
import configs.configs_v2 as configs

#evaluation
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes

# model

#env
import env
import os
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (6,6)
# matplotlib inline


tf = try_import_tf()
tfp = try_import_tfp()

if type(tf) == tuple:
    tf = tf[0]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DebuggingLayersMixin:
    def __init__(self):
        self.compute_encoding_layer = make_tf_callable(self.get_session())(self.model.encoder_output)
        self.compute_action = make_tf_callable(self.get_session())(self.model.action_computation)# DO THIS
        self.output_inputs = make_tf_callable(self.get_session())(self.model.output_inputs)

def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    DebuggingLayersMixin.__init__(policy)


MixinPPOTFPolicy = PPOTFPolicy.with_updates(
    name="MixinPPOTFPolicy",
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, DebuggingLayersMixin
    ])

def get_policy_class(config):
    return MixinPPOTFPolicy


MixinPPOTrainer = PPOTrainer.with_updates(
    name="MixinPPOTrainer",
    default_policy=MixinPPOTFPolicy,
    get_policy_class=get_policy_class,

)

from policies.heuristic_handcrafted_policy import HeuristicHandcraftedTrainer

def generate_checkpoint_dir(heuristic_handcrafted, policy_dir, env_name, experiment, ncheckpoint): ## TODO!!!!
    savedir = './ray_results'
    if heuristic_handcrafted:
        if policy_dir != '':
            savedir = savedir + '/' + policy_dir
        else:
            savedir = savedir + '/' + env_name
        # if not os.path.isdir(savedir):
        #    os.makedirs(savedir)
        experiment = 1
        expstr = '/exp' + str(experiment)
        video_dir = savedir + '/videos_exp_' + str(
            1)  # _' + str(experiment) + '_ckpoint_' + str(ncheckpoint)
        if not os.path.isdir(video_dir): os.makedirs(video_dir)
        savedir = savedir + expstr
        savedir = savedir + '/' + os.listdir(savedir)[0]
        if 'Trainer' in os.listdir(savedir)[0]:
            savedir = savedir + '/' + os.listdir(savedir)[0]
        else:
            savedir = savedir + '/' + os.listdir(savedir)[1]
        savedir = savedir + '/checkpoint_' + str(1) + '/checkpoint-' + str(1)
        checkpoint = savedir
    else:
        if policy_dir != '':
            savedir = savedir + '/' + policy_dir
        else:
            savedir = savedir + '/' + env_name
        # if not os.path.isdir(savedir):
        #    os.makedirs(savedir)
        experiment = experiment
        expstr = '/exp' + str(experiment)
        video_dir = savedir + '/videos_exp_' + str(experiment) + '_ckpoint_' + str(ncheckpoint)
        if not os.path.isdir(video_dir): os.makedirs(video_dir)
        savedir = savedir + expstr
        savedir = savedir + '/' + os.listdir(savedir)[0]
        if 'Trainer' in os.listdir(savedir)[0]:
            savedir = savedir + '/' + os.listdir(savedir)[0]
        else:
            savedir = savedir + '/' + os.listdir(savedir)[1]
        savedir = savedir + '/checkpoint_' + str(ncheckpoint) + '/checkpoint-' + str(ncheckpoint)
        checkpoint = savedir
    return checkpoint, video_dir

def main(args):
    ray.init(local_mode=True)#, redis_max_memory=int(6e9))

    # Environment variables/creator
    config_env = configs.config_env(args)
    config_env["nclasses"] = 2
    if args.env == 'SceneEnv_RLlibMA_test_airsim':
        from env.SceneEnv_RLlibMA_test_airsim import SceneEnv
        config_env["heuristic_policy"] = args.heuristic_policy
        config_env["heuristic_target_order"] = args.heuristic_target_order
        config_env["reverse_heuristic_target_order"] = args.reverse_heuristic_target_order
        # config_env["static_targets"] = args.static_targets
        config_env["test"] = args.test
        config_env["env_mode"] = args.env_mode
        config_env["save_scn"] = args.save_test_scn
        config_env["save_scn_folder"] = args.test_save_dir
        config_env["max_episodes"] = args.episodes
        config_env['load_scn'] = args.load_test_scn
        config_env['load_scn_folder'] = args.test_load_dir
        config_env['reward_1target'] = False
        config_env['horizon'] = args.horizon
        config_env['random_beliefs'] = False
        config_env['random_static_dynamic'] = True
        from utils.yolo_model import create_trained_yolo
        import env.settings as settings
        settings.init()
        settings.yolo = create_trained_yolo()
        env = SceneEnv(config_env)
        register_env("SceneEnv", lambda c: SceneEnv(config_env))
        env = SceneEnv(config_env)

    # Training and model configuration
    config, stop = configs.config(args, env)
    config['env']='SceneEnv'
    config['horizon'] = args.horizon  # 40 # TODO MAX steps come here
    config["explore"] = False


    # Model configuration

    config["model"]= {
            #"fcnet_hiddens": [args.num_units] * 2,
            #"fcnet_activation": "relu",
            "custom_model": "customized_model",
            "custom_model_config": {
                "num_other_robots": env.nrobots-1, # this shall be DEPRECATED in the FUTURE (need to compute this online)
                "num_targets": env.MAX_TARGETS,#env.ntargets,
                # "dim_p": env.observation_space_dict[0].spaces[0]["targets"]["all"].child_space['location'].shape[0]
                #         if env.multiagent_policy else env.observation_space["targets"]["all"].child_space['location'].shape[0], #env.observation_space.child_space['location'].shape[0],
                "dim_p": env.observation_space_dict[0].child_space['location'].shape[0]
                        if env.multiagent_policy else env.observation_space.child_space['location'].shape[0], #env.observation_space.child_space['location'].shape[0],
                "training": True
            }
        }
    config["vf_share_layers"] = True
    config["explore"] = False

    config['lr'] =3e-4# 1e4 #9.99999e5
    config['num_workers'] = 0
    config['no_done_at_end']=True


    ### MULTIAGENT ADAPTATION ###
    def gen_policy(i):
        return (
            None,
            env.observation_space_dict[i],
            env.action_space_dict[i],
            {
                "agent_id": i,
                "obs_space_dict": env.observation_space_dict[i],
                "act_space_dict": env.action_space_dict[i],
            }
        )

    policies = {"policy_0": gen_policy(0)}
    policy_ids = ["policy_0"]
    if env.multiagent_policy:
        config["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn": ray.tune.function(
                lambda i: policy_ids[0],
            )
        }

    model_paths = []
    env.reset()
    env_name = "SceneEnv"

    ### MODELS ###
    ## Set Transformers ## PLOT SEED 3 OR 5
    from models.lee_setTransformers_opt_v2 import SetTransformers as loaded_model  # Basic implementation: SAB + PMA BEST UNTIL THE MOMENT
    policy_dir = "CORL2022/our_method_airper/seed100/2ndphase_v2"
    experiment = 1
    ncheckpoint = 5750
    checkpoint_dir, video_dir = generate_checkpoint_dir(args.heuristic_handcrafted, policy_dir, env_name, experiment, ncheckpoint)
    model_paths += [
       checkpoint_dir
       ]

    ## Single-target
    # from models.baseline2 import FullyConnectedModel as loaded_model
    # policy_dir = "sim_per/baseline2_v2/seed100"
    # experiment = 1
    # ncheckpoint = 6000
    # checkpoint_dir, video_dir = generate_checkpoint_dir(args.heuristic_handcrafted, policy_dir, env_name, experiment, ncheckpoint)
    # model_paths += [
    #     checkpoint_dir
    #     ]

    ## Baseline 1
    # policy_dir = "CORL2022/baseline_1"
    # experiment = 1
    # ncheckpoint = 1
    # checkpoint_dir, video_dir = generate_checkpoint_dir(args.heuristic_handcrafted, policy_dir, env_name, experiment, ncheckpoint)

    if not args.heuristic_handcrafted:
        ModelCatalog.register_custom_model("customized_model", loaded_model)  # From v8 onwards
        agent = MixinPPOTrainer(config=config, env="SceneEnv")
        agent.restore(checkpoint_path=model_paths[0])
    else:
        from ray.rllib.agents import with_common_config
        DEFAULT_CONFIG = with_common_config({})
        DEFAULT_CONFIG['env_config']['MAXTARGETS'] = args.ntargets[0]
        agent = HeuristicHandcraftedTrainer(config = DEFAULT_CONFIG,env="SceneEnv")


    ## allow logs
    env.log_folder = video_dir
    env.logs = True

### environment TESTS!!
    print('Init Scene')

    # plot_scene = True
    # if plot_scene:
    #     fig, ax = env.init_plot_scene(num_figure=1)
    #     # camera = Camera(fig)
    #
    # plot_beliefs = True
    # if plot_beliefs:
    #     fig2, ax2 = env.init_plot_beliefs(num_figure=2)
    #
    # plot_image = False
    # if plot_image:
    #     fig3, ax3 = env.init_plot_images(num_figure=3)
    #
    # if plot_image or plot_beliefs or plot_scene:
    #     env.render_enabled = True
    #     plt.show()

    # # MAIN LOOP
    final = False
    taux = 0

    obs = env.reset()
    # env.render()
    action = agent.compute_action(obs[0])
    print("everything's peachy")

    # # MAIN LOOP
    final = False
    taux = 0
    # we ran one full episode with random actions to check that everything works
    # """
    auxvartime = time.time()
    nepisodes = 0
    episode_duration = []
    taux_ep = time.time()
    episode_info = []
    episode_info.append({"timestamp": [], "beliefs": []})
    episode_info[nepisodes]["timestamp"].append(time.time()-taux_ep)
    episode_info[nepisodes]["beliefs"].append(env.beliefs[0])
    while nepisodes < 10:
        # print("Pipeline time:",time.time()-auxvartime)
        auxvartime=time.time()
        action = {'0': agent.compute_action(obs[0])}
        obs, reward, final, _ = env.step(action)  # , envtest = True)
        episode_info[nepisodes]["timestamp"].append(time.time() - taux_ep)
        episode_info[nepisodes]["beliefs"].append(env.beliefs[0].copy())
        # env.render()

        # # Update plots of the scene
        # if plot_scene:
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()
        #
        # # Update plot of the image
        # if plot_image:
        #     fig3.canvas.draw()
        #     fig3.canvas.flush_events()
        #
        # # Update plots of beliefs
        # if plot_beliefs:
        #     fig2.canvas.draw()
        #     fig2.canvas.flush_events()

        #print(final[0])
        if (taux % 400 == 0 and taux != 0) or final[0]:
            episode_duration.append(time.time()-taux_ep)
            taux = 0
            print("episode time:", time.time()-taux_ep)
            taux_ep = time.time()
            nepisodes +=1
            episode_info.append({"timestamp": [], "beliefs": []})
            print("episode:",nepisodes-1)
            env.reset()
        else:
            taux += 1
    import pickle
    toprint = pickle.dumps(episode_info)
    print(toprint)
    with open('file_ral_videos2.pkl','wb') as file:
        pickle.dump(episode_info, file)


def beliefs2video(beliefs_episodes_array = []):
    ntargets = 20
    number_of_classes = 2
    episode = 0
    target_colors = np.random.rand(3, ntargets)
    beliefs_array = beliefs_episodes_array[episode]['beliefs']
    timestamp_array = beliefs_episodes_array[episode]['timestamp']

    beliefs = beliefs_array[0]
    bar_plots = []
    import math
    import matplotlib.gridspec as gridspec


    robot_plot_belief = 0
    dim_window = int(np.ceil(math.sqrt(ntargets)))
    tags = []
    for classes in range(number_of_classes):
        label = 'L' + str(classes)
        tags = np.append(tags, label)
    y_pos = np.arange(len(tags))

    fig = plt.figure()
    outer = gridspec.GridSpec(dim_window, dim_window, wspace=0.1, hspace=0.1)
    ax = []
    for t in range(ntargets):
        ax.append(plt.Subplot(fig, outer[t]))
        ax[t].set_xlim((-0.75, number_of_classes - 1 + 0.75))
        ax[t].set_ylim((0, 1))
        if t >= ntargets - dim_window:
            ax[t].set_xticks(y_pos)  # values
            ax[t].set_xticklabels(tags)  # labels
        else:
            ax[t].set_xticks([])
        ax[t].set_yticks([])
        bar_plot = ax[t].bar(tags, beliefs[t, :], color=target_colors[:, t])
        bar_plots.append(bar_plot)
        fig.add_subplot(ax[t])

    taux = time.time()
    taux2 = taux
    init_ep = 1
    for episode in range(len(beliefs_episodes_array)):
        beliefs_array = beliefs_episodes_array[episode]['beliefs']
        timestamp_array = beliefs_episodes_array[episode]['timestamp']
        for i in range(init_ep,len(beliefs_array)):
            if i>0:
                time.sleep(timestamp_array[i] - timestamp_array[i - 1] - (time.time() - taux2))
            beliefs = beliefs_array[i]
            for t in range(ntargets):
                for c in range(number_of_classes):
                    bar_plots[t][c].set_height(beliefs[t, c])
            fig.canvas.draw()
            fig.canvas.flush_events()
            taux2 = time.time()
        print('time:', time.time()-taux)
        taux = time.time()
        taux2 = taux


if __name__ == '__main__':
    args = parse_args()
    args.ntargets=[20]
    args.heuristic_handcrafted = False
    args.heuristic_policy = False or args.heuristic_handcrafted




    args.heuristic_target_order = False or args.heuristic_policy or args.heuristic_handcrafted
    args.test=True
    main(args)

    # import pickle
    # with open('file_ral_videos2.pkl','rb') as file:
    #     beliefs = pickle.load(file)
    #
    # beliefs2video(beliefs_episodes_array = beliefs)