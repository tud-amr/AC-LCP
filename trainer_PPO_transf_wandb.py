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
# parser
from utils.parse_args import parse_args
# configs
import configs.configs_v2 as configs
import wandb

# model

#env
import os

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

def main(args):
    ray.init(local_mode=False)#, redis_max_memory=int(6e9))

    # Model configuration
    ## Deep Sets baselines
    # from models.deepsets_decoder import AttentionDeepSets as loaded_model

    ## LSTM baseline
    # from models.lstm_encoder import LSTM_Encoder as loaded_model
    # args.reverse_heuristic_target_order = True

    # Set transformers -- Lee et al. original model and adaptations
    from models.multi_target import SetTransformers as loaded_model  # Basic implementation: SAB + PMA BEST UNTIL THE MOMENT

    # Baseline 2
    # from models.single_target import FullyConnectedModel as loaded_model

    # Environment variables/creator
    config_env = configs.config_env(args)


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
        config_env['random_beliefs'] = True
        config_env['random_static_dynamic'] = True
        register_env("SceneEnv", lambda c: SceneEnv(config_env))
        env = SceneEnv(config_env)
    elif args.env == 'SceneEnv_RLlibMA_simper':
        from env.SceneEnv_RLlibMA_simper import SceneEnv
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
        config_env['random_beliefs'] = True
        config_env['random_static_dynamic'] = True
        register_env("SceneEnv", lambda c: SceneEnv(config_env))
        env = SceneEnv(config_env)

    # Training and model configuration
    config, stop = configs.config(args, env)
    config['env']='SceneEnv'
    config['horizon'] = args.horizon  # 40 # TODO MAX steps come here


    # Register custom model into the ray framework
    ModelCatalog.register_custom_model("customized_model", loaded_model)  # From v8 onwards
    config["model"]= {
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
    config["vf_share_layers"] = False

    Trainer = MixinPPOTrainer
    """ SMALL PATCH TO GET A DUMMY CONFIG FOR AN HEURISTIC POLICY
    if args.policy_dir == 'heuristic_policy_config':
        from policies.heuristic_handcrafted_policy import HeuristicHandcraftedTrainer
        Trainer = HeuristicHandcraftedTrainer
        args.checkpoint_freq = 1
        from ray.rllib.agents import with_common_config
        DEFAULT_CONFIG = with_common_config({})
        config = DEFAULT_CONFIG
        config['env'] = 'SceneEnv'
    #"""

    savedir = './ray_results'
    if args.policy_dir != '':
        savedir =  savedir + '/' + args.policy_dir
    else:
        savedir = savedir + '/' + args.env
    experiment=1
    expstr = '/exp'+str(experiment)
    while os.path.isdir(savedir+expstr):
        experiment += 1
        expstr = '/exp'+str(experiment)
    savedir = savedir+expstr

    config["num_gpus"] = 1
    config['lr'] = 3e-4# 1e4 #9.99999e5
    config['num_workers'] = 10
    config['no_done_at_end'] = False
    config["rollout_fragment_length"] = 1600

    # Run training
    from ray.tune.logger import pretty_print
    trainer = Trainer(config=config,env="SceneEnv")
    if args.restore is not None:
        trainer.restore(checkpoint_path=args.restore)
    wandb.init(project="TESTING_BEFORE_PUBLISHING", config=config, name="trained_model")

    # for iteration in range(3750):
    for iteration in range(6000):
        results = trainer.train()
        results.pop("config")
        wandb.log(results)
        print(pretty_print(results))
        if iteration%500 == 0:
            checkpoint_dir = trainer.save(checkpoint_dir=savedir)
            print(f"Checkpoint saved in directory {checkpoint_dir}")


    checkpoint_dir = trainer.save(checkpoint_dir=savedir)
    print(f"Checkpoint saved in directory {checkpoint_dir}")
    ray.shutdown()


if __name__ == '__main__':
    args = parse_args()
    # args.restore = "ray_results/RAL2023/Ours/setTransformer_opt_v2/50x50_env/seed100/2ndphase/exp1/MixinPPOTrainer/MixinPPOTrainer_SceneEnv_0_2022-04-06_13-04-49a0r8zsqv/checkpoint_8000"
    main(args)
