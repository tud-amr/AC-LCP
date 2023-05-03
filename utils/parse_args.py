import argparse


def parse_args():
    parser = argparse.ArgumentParser("Ray training with custom IG environment")


    # save and restore file management
    parser.add_argument(
        "--policy-dir", type=str, help="folder name of the policy.", default="")
    parser.add_argument(
        "--experiment", type=str, help="chosen experiment to reload.", default="")
    parser.add_argument(
        "--ncheckpoint", type=str, help="chosen checkpoint to reload.", default="")

    # Checkpoint
    parser.add_argument(
        "--checkpoint", type=str, help="Checkpoint from which to roll out.", default="")

    parser.add_argument(
        "--run",
        type=str,
        required=False,
        default="",
        help="The algorithm or model to train. This may refer to the name "
             "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
             "user-defined trainable function or class registered in the "
             "tune registry.")
    parser.add_argument(
        "--env", type=str, help="The gym environment to use.")

    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Suppress rendering of the environment.")

    parser.add_argument(
        "--episodes",
        default=0,
        help="Number of complete episodes to roll out (overrides --steps).")

    ### Old arguments needs a cleanup

    parser.add_argument("--scenario", type=str, default="simple_spread_assigned",
                        choices=['simple', 'simple_speaker_listener',
                                 'simple_crypto', 'simple_push',
                                 'simple_tag', 'simple_spread', 'simple_adversary', 'simple_spread_assigned',
                                 'matlab_simple_spread_assigned','matlab_simple_spread_assigned_hardcoll', 'matlab_simple_spread_assigned_checkpoints'],
                        help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100,
                        help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000,
                        help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0,
                        help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg",
                        help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg",
                        help="policy of adversaries")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    # NOTE: 1 iteration = sample_batch_size * num_workers timesteps * num_envs_per_worker
    parser.add_argument("--sample-batch-size", type=int, default=25,
                        help="number of data points sampled /update /worker")
    parser.add_argument("--train-batch-size", type=int, default=1024,
                        help="number of data points /update")
    parser.add_argument("--n-step", type=int, default=1,
                        help="length of multistep value backup")
    parser.add_argument("--num-units", type=int, default=128,
                        help="number of units in the mlp")
    parser.add_argument("--replay-buffer", type=int, default=1000000,
                        help="size of replay buffer in training")
    parser.add_argument("--seed", type=int, default=100,
                        help="initialization seed for the network weights")

    # Checkpoint
    parser.add_argument("--checkpoint-freq", type=int, default = 100, #75,
                        help="save model once every time this many iterations are completed")
    parser.add_argument("--local-dir", type=str, default="./ray_results",
                        help="path to save checkpoints")
    parser.add_argument("--restore", type=str, default=None,
                        help="directory in which training state and model are loaded")
    parser.add_argument("--in-evaluation", type=bool, default=False, help="trigger evaluation procedure")

    # Parallelism
    #parser.add_argument("--num-workers", type=int, default=0)
    #parser.add_argument("--num-envs-per-worker", type=int, default=1)
    #parser.add_argument("--num-gpus", type=int, default=0)

    parser.add_argument("--num-workers", type=int, default=0)  #0
    parser.add_argument("--num-envs-per-worker", type=int, default=1)  #1
    parser.add_argument("--num-gpus", type=int, default=0)  #0
    #parser.add_argument("--num-cpus-per-worker", type=int, default=1)
    parser.add_argument("--num-gpus-per-worker", type=int, default=0)  #0

    # From the ppo
    parser.add_argument("--stop-iters", type=int, default=100)
    parser.add_argument("--stop-timesteps", type=int, default=160000000)
    # parser.add_argument("--stop-reward", type=float, default=7.99)

    # For rollouts
    parser.add_argument("--stop-iters-rollout", type=int, default=1)
    parser.add_argument("--nagents", type=int, default=1)
    parser.add_argument("--ntargets", type=int, nargs='+', default=3)
    parser.add_argument("--nrobots", type=int, nargs='+', default=1)
    parser.add_argument("--horizon", type=int, default=40)

    # mode of hand-engineered comm. policy (-1 no hand-engineered)
    parser.add_argument("--mode", type=int, default=-1)
    parser.add_argument("--test", type=int, default=0, choices = [0,1], help="whether we want to test the policy or not")
    #parser.add_argument("--test-env", type=int, default=0, choices = [0,1], help="whether we want to act in the test environment or not")
    parser.add_argument("--deterministic", type=int, default=1, choices=[0, 1],
                        help="enable exploration or not during execution")
    parser.add_argument("--heuristic-policy", const=True, action="store_const", default=False, help="whether we want to use the heuristic policy or not")
    parser.add_argument("--heuristic-target-order", const=True, action="store_const", default=False,
                        help="whether we want to use the heuristic policy or not")
    parser.add_argument("--reverse-heuristic-target-order", const=True, action="store_const", default=False,
                        help="whether we want to use the heuristic policy or not")

    parser.add_argument("--env-mode", type=str,required=False, default="static",
                        choices=['static', 'brown',
                                 'cte_vel', 'sf_goal','airsim'],
                        help="environment dynamics")
    parser.add_argument("--save-test-scn", const=True, action="store_const", default=False,
                        help="save scenarios - only under evaluation")
    parser.add_argument("--load-test-scn", const=True, action="store_const", default=False,
                        help="load scenarios - only under evaluation")
    parser.add_argument("--test_save_dir", type=str, required=False, default="",
                        help="folder where saved test scenarios reside")
    parser.add_argument("--test_load_dir", type=str, required=False, default="",help="folder where loaded test scenarios reside")

    # finish training
    parser.add_argument("--training-iteration", type=int, default=11000)

    return parser.parse_args()