## New trainings in new perception setting
# Our method
# python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA_reviewers --policy-dir RAL2022/our_method_PhREnv/seed100/1stphase --nrobots 1 --ntargets 1 12 --training-iteration 6000 --env-mode cte_vel --horizon 400 --seed 100
# Lstm Encoder
# python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA_reviewers --policy-dir RAL2022/LSTM_PhREnv/seed100/1stphase --nrobots 1 --ntargets 1 12 --training-iteration 6000 --env-mode cte_vel --horizon 400 --seed 100 --reverse-heuristic-target-order
# Baseline 2
# python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA_reviewers --policy-dir RAL2022/baseline2_PhREnv/seed100 --nrobots 1 --ntargets 1 --training-iteration 6000 --env-mode cte_vel --heuristic-policy --horizon 400 --seed 100
# DeepSets Decoder
# python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA_reviewers --policy-dir RAL2022/deepsets_PhREnv/seed100/1stphase --nrobots 1 --ntargets 1 12 --training-iteration 6000 --env-mode cte_vel --horizon 400 --seed 100


# Training scripts
# Simulated perception environment
# Multi-target(ours) 1st and 2nd phase
python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA --policy-dir RAL2022/our_method/seed100/2ndphase --nrobots 1 --ntargets 1 12 --training-iteration 6000 --env-mode cte_vel --horizon 400 --seed 100
python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA --policy-dir RAL2022/our_method/seed100/2ndphase --nrobots 1 --ntargets 1 6 --training-iteration 8000 --env-mode cte_vel --horizon 400 --seed 100
# LSTM encoder 1st and 2nd phase
python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA --policy-dir RAL2022/lstm/seed100/2ndphase --nrobots 1 --ntargets 1 12 --training-iteration 6000 --env-mode cte_vel --horizon 400 --seed 100 --reverse-heuristic-target-order
python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA --policy-dir RAL2022/lstm/seed100/2ndphase --nrobots 1 --ntargets 1 6 --training-iteration 8000 --env-mode cte_vel --horizon 400 --seed 100 --reverse-heuristic-target-order
# DeepSets decoder 1st and 2nd phase
python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA --policy-dir RAL2022/deepsets/seed100/2ndphase --nrobots 1 --ntargets 1 12 --training-iteration 6000 --env-mode cte_vel --horizon 400 --seed 100
python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA --policy-dir RAL2022/deepsets/seed100/2ndphase --nrobots 1 --ntargets 1 6 --training-iteration 8000 --env-mode cte_vel --horizon 400 --seed 100
# Single-target (ours) 1st and 2nd phase
python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA --policy-dir RAL2022/baseline2_PhREnv/seed100 --nrobots 1 --ntargets 1 --training-iteration 6000 --env-mode cte_vel --heuristic-policy --horizon 400 --seed 100
python trainer_PPO_transf_wandb.py --env SceneEnv_RLlibMA --policy-dir RAL2022/baseline2_PhREnv/seed100 --nrobots 1 --ntargets 1 --training-iteration 8000 --env-mode cte_vel --heuristic-policy --horizon 400 --seed 100

