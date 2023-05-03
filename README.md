# Active Classification of Moving Targets with Learned Control Policies


This repository contains the code and data for the paper titled "Active Classification of Moving Targets with Learned Control Policies" published in IEEE Robotics and Automation Letters. We are in the process of cleaning up and documenting the code and will keep maintaining and updating the repository, clarifying and documenting how to install key features employed in the paper.


## Installation

First create a virtual environment with python 3.6 and activate it.
```bash
conda create --name venv python=3.6
conda activate venv
```
Then install the required packages using
```bash
pip install -r requirements.txt
```

## Policy training
For training the policy's first phase, run
```bash
python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA --policy-dir RAL2023/our_method/seed100/1stphase --nrobots 1 --ntargets 1 12 --training-iteration 6000 --env-mode cte_vel --horizon 400 --seed 100
```
For training the policy's second phase, give the address of the pretrained weights to args.restore and run
```bash
python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA --policy-dir RAL2023/our_method/seed100/2ndphase --nrobots 1 --ntargets 1 6 --training-iteration 8000 --env-mode cte_vel --horizon 400 --seed 100
```

## Policy evaluation
For evaluating the policy's second phase, run
```bash
python policy_evaluation --env SceneEnv_RLlibMA_test --env-mode sf_goal --horizon 400
```

## Trouble shooting
If you get the following error
```bash
File "/home/amr/miniconda3/envs/reproenv/lib/python3.6/site-packages/ray/rllib/utils/spaces/repeated.py", line 20, in __init__
    self.np_random = np.random.RandomState()
AttributeError: can't set attribute
```
then you need to go to the file
```bash
/home/amr/miniconda3/envs/reproenv/lib/python3.6/site-packages/ray/rllib/utils/spaces/repeated.py
```
and comment lines 20, 25, 26 and 27


## Citation
If you find this code useful in your research, please consider citing:
```bash
@ARTICLE{10111041,
  author={Serra-Gómez, Álvaro and Montijano, Eduardo and Böhmer, Wendelin and Alonso-Mora, Javier},
  journal={IEEE Robotics and Automation Letters}, 
  title={Active Classification of Moving Targets with Learned Control Policies}, 
  year={2023},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/LRA.2023.3271508}}
```
