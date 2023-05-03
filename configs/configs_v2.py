import ray
from utils.callbacks import MyCallbacks


def config_env(args):
    config_mpe = {
        "dimensions": 3,
        "ntargets": args.ntargets,
        "nrobots": args.nrobots,
        "nclasses": 6,
        "MAXTARGETS": 8,
        "horizon": 40
    }
    return config_mpe


def config(args, env):

    # Definition of Policies
    def gen_policy(i):
        return (
            None,
            env.observation_space,
            env.action_space,
            {
                "agent_id": 0,
                "obs_space_dict": {0:env.observation_space},
                "act_space_dict": {0:env.action_space},
            }
        )


    policies = {"policy_0": gen_policy(0)} #{"policy_%d" % i: gen_policy(i) for i in range(len(env.observation_space_dict))}
    policy_ids = ["policy_0"]  # list(policies.keys())

    config = {
        # training seed
        "seed": args.seed, # 100 ONLY FOR EVALUATION

        # environment
        "env": "mpe",
        "env_config": config_env(args),

        # ray
        "num_envs_per_worker": args.num_envs_per_worker,
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,
        #"num_cpus_per_worker": args.num_cpus_per_worker,
        "num_gpus_per_worker": args.num_gpus_per_worker,

        "callbacks": MyCallbacks,

        # evaluation
        "in_evaluation": args.in_evaluation,

        ### Training ###
        ## Number of timesteps collected for each SGD round. This defines the size
        ## of each SGD epoch.
        "train_batch_size": 16000, #4000,#48000, #args.nagents*100*40,#48000, #4000,  # 2000,
        ## Stepsize of SGD (learning rate).
        "lr": 0.0003,#5e-5,  # 2.5e-4,
        ## Default exploration behavior, iff `explore`=None is passed into
        ## compute_action(s).
        ## Set to False for no exploration behavior (e.g., for evaluation).
        "explore": True,


        # ppo training
        # "batch_mode": "complete_episodes",

        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE(lambda) parameter.
        "lambda": 0.95,#1.0, #0.95,  # 0.99,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,#1.0,  # 0.2,


        # Whether to clip rewards prior to experience postprocessing. Setting to
        # None means clip for Atari only.
        "clip_rewards": None,#False,

        # Size of batches collected from each worker.
        "rollout_fragment_length": 800,#200, #2400,#args.nagents*100*2,#2400, #200,#20,  # 5,

        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 256,#512 #128,#256,  # 128,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30, #20,  # 10,

        # Learning rate schedule.
        "lr_schedule": None,
        # Share layers for value function. If you set this to True, it's important
        # to tune vf_loss_coeff.
        "vf_share_layers": False,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers: True.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.001,#0.01, # in standard config its a 0 but better not consider it
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.3,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param":  10,#50,  # 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": 0.1, #None
        # Target value for KL divergence.
        "kl_target": 0.01, #15e-4,  # 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes" to
        # `rollout_fragment_length` length unrolls. Episode truncation guarantees
        # evenly sized batches, but increases variance as the reward-to-go will
        # need to be estimated at truncation boundaries.
        "batch_mode": "truncate_episodes",  # "complete_episodes"
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # Uses the sync samples optimizer instead of the multi-gpu one. This is
        # usually slower, but you might want to try it if you run into issues with
        # the default optimizer.
        "simple_optimizer": False,
        # Whether to fake GPUs (using CPUs).
        # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
        "_fake_gpus": False,

        "gamma": 0.99,
        # Number of steps after which the episode is forced to terminate. Defaults
        # to `env.spec.max_episode_steps` (if present) for Gym envs.
        "horizon": args.max_episode_len,
        # "lr": 1e-2,

        #"multiagent": {
        #    "policies": policies,
        #    "policy_mapping_fn": ray.tune.function(
        #        lambda i: policy_ids[i],
        #    )
        #},
        "model": {
            "fcnet_hiddens": None,
            "fcnet_activation": None,
            # "custom_model": "cc_model",
        },
        "framework": "tf",
    }


    stop = {
        "training_iteration": args.training_iteration,#11000,#6000,#313*3,
        #"timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
    }

    if args.in_evaluation:
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions.
        # IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
        # policy, even if this is a stochastic one. Setting "explore=False" here
        # will result in the evaluation workers not using this optimal policy!
        config["evaluation_config"] = {
            # Example: overriding env_config, exploration, etc:
            # "env_config": {...},
            "explore": False,
            "train_batch_size": 12000, #10000,
            "lr": 0.0,
        }
        ### Evaluation ###
        print("IN EVALUATION CONFIG!!")
        config["explore"] = False
        config["train_batch_size"] = 120000
        config["lr"] = 0.0

    return config, stop