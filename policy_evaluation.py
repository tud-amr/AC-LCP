import ray
from ray.tune.registry import register_env
# parser
from utils.parse_args import parse_args

# callbacks

# configs
import configs.configs_v2 as configs

#evaluation
import wandb

#env
import os

import matplotlib.pyplot as plt
# matplotlib inline

from policies.heuristic_handcrafted_policy import HeuristicHandcraftedTrainer

# GAE slight modif. policy
import ray
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf, try_import_tfp
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.agents.ppo.ppo_tf_policy import ValueNetworkMixin, KLCoeffMixin
from ray.rllib.utils.tf_ops import make_tf_callable

###########################################################################################
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

from custom_ray.ppo_functions import postprocess_ppo_gae
MixinPPOTFPolicy = PPOTFPolicy.with_updates(
    name="MixinPPOTFPolicy",
    before_loss_init=setup_mixins,
    postprocess_fn=postprocess_ppo_gae,
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


##############################

def generate_checkpoint_dir(heuristic_handcrafted, policy_dir, env_name, experiment, ncheckpoint, old_system=True): ## TODO!!!!
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
        if old_system:
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
        if old_system:
            savedir = savedir + '/' + os.listdir(savedir)[0]
            if 'Trainer' in os.listdir(savedir)[0]:
                savedir = savedir + '/' + os.listdir(savedir)[0]
            else:
                savedir = savedir + '/' + os.listdir(savedir)[1]
        savedir = savedir + '/checkpoint_' + str(ncheckpoint) + '/checkpoint-' + str(ncheckpoint)
        checkpoint = savedir
    return checkpoint, video_dir

def load_config_env_param(args):
    # Import custom models
    from models.multi_target import SetTransformers
    from models.single_target import FullyConnectedModel
    # Register custom model into the ray framework
    # Tf
    ModelCatalog.register_custom_model("Set_Transformers", SetTransformers)
    ModelCatalog.register_custom_model("baseline2", FullyConnectedModel)

    # Environment variables/creator
    config_env = configs.config_env(args)
    config_env["nclasses"] = 2
    if args.env == 'SceneEnv_RLlibMA':
        from env.SceneEnv_RLlibMA import SceneEnv
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
        register_env("SceneEnv", lambda c: SceneEnv(config_env))
        env = SceneEnv(config_env)
    elif args.env == 'SceneEnv_RLlibMA_test':
        from env.SceneEnv_RLlibMA_test import SceneEnv
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
        register_env("SceneEnv", lambda c: SceneEnv(config_env))
        env = SceneEnv(config_env)

    # Training and model configuration
    config, stop = configs.config(args, env)

    config['env'] = 'SceneEnv'
    config['horizon'] = args.horizon  # 40 # TODO MAX steps come here
    config["model"] = {
        "custom_model": None,  # "SE_Attention",  # "customized_model",
        "custom_model_config": {
            "num_other_robots": env.nrobots - 1,  # this shall be DEPRECATED in the FUTURE (need to compute this online)
            "num_targets": env.MAX_TARGETS,  # env.ntargets,
            "dim_p": env.observation_space_dict[0].spaces[0].child_space['location'].shape[0]
            if env.multiagent_policy else env.observation_space.child_space['location'].shape[0],
            "training": True
        }
    }
    config["explore"] = False

    # Final training specifications
    config['no_done_at_end'] = True
    config['lr'] = 3e-4  # 1e4 #9.99999e5
    config["num_gpus"] = 0
    config["num_workers"] = 0
    config["grad_clip"] = 0.1
    config["rollout_fragment_length"] = 2000
    config["vf_loss_coeff"] = 0.5
    config["vf_clip_param"] = 120
    config["model"]["custom_model_config"]["num_gpus"] = config["num_gpus"]
    config["model"]["custom_model_config"]["vf_share_layers"] = False

    env_name = "SceneEnv"
    return env, env_name, config

def load_agent(args,env_name,config):
    model_paths = []
    # Model architecture and registering
    ### Tensorflow policies
    ## OURS
    policy_dir = "RAL2023/Ours/setTransformer_opt_v2/50x50_env/seed100/2ndphase"
    config["framework"] = "tf"
    config["model"]["custom_model"] = "Set_Transformers"
    experiment = 1
    ncheckpoint = 8000
    checkpoint_dir, video_dir = generate_checkpoint_dir(args.heuristic_handcrafted, policy_dir, env_name, experiment,
                                                        ncheckpoint, old_system=True)
    model_paths += [
        checkpoint_dir
    ]

    ## Single-target
    # policy_dir = "CORL2022/baseline_2/seed100"
    # config["model"]["custom_model"] = "baseline2"
    # config["framework"] = "tf"
    # experiment = 1
    # ncheckpoint = 6000
    # checkpoint_dir, video_dir = generate_checkpoint_dir(args.heuristic_handcrafted, policy_dir, env_name, experiment,
    #                                                     ncheckpoint, old_system=True)
    # model_paths += [
    #     checkpoint_dir
    # ]

    #Trainer = PPOTrainer
    Trainer = MixinPPOTrainer
    agent = Trainer(config=config, env="SceneEnv")
    agent.restore(checkpoint_path=model_paths[0])
    return agent, video_dir

def print_additional_info(obs):
    pass

def main(args):
    ray.init(local_mode=True)#, redis_max_memory=int(6e9))
    env, env_name, config = load_config_env_param(args)
    env.reset()
    agent, video_dir = load_agent(args,env_name,config)

    ## allow logs
    env.log_folder = video_dir
    env.logs = True

    ## MAIN LOOP
    obs = env.reset()
    #action = agent.compute_action(obs[0], policy_id=list(agent.config['multiagent']['policies'].keys())[0] if env.multiagent_policy else "default_policy")
    # take the first key from a dict



    print("everything's peachy")
    # # MAIN LOOP
    final = False
    taux = 0
    # we ran one full episode with random actions to check that everything works
    # """
    nepisodes = 0
    while nepisodes < 50:
        # action = {'0': agent.compute_action(obs[0]), '1':agent.compute_action(obs[1])}
        action = {}
        for r in range(env.nrobots):
            action[str(r)] = agent.compute_action(obs[r], policy_id=list(agent.config['multiagent']['policies'].keys())[0] if env.multiagent_policy else "default_policy")
        obs, reward, final, info = env.step(action)  # , envtest = True)
        env.render()

        if (taux % 400 == 0 and taux != 0) or final[0]:
            taux = 0
            nepisodes +=1
            print("episode:",nepisodes-1)
            env.reset()
        else:
            taux += 1


    ray.shutdown()

if __name__ == '__main__':
    args = parse_args()
    mode = "main"
    if mode == "main":
        nrobots = [1]
        args.nrobots = nrobots
        ntargetsList = [20]
        for ntargets in ntargetsList:
            args.ntargets=[ntargets]
            args.heuristic_handcrafted = False
            args.heuristic_policy = False or args.heuristic_handcrafted

            args.heuristic_target_order = False or args.heuristic_policy or args.heuristic_handcrafted
            args.test=True
            main(args)