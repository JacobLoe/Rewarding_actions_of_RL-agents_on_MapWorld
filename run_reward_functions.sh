#!/bin/bash
python main.py random --parameters parameters/reward_function_r5.json --base_path results/random/2022-01-23_r5
echo " "
python main.py random --parameters parameters/reward_function_r6.json --base_path results/random/2022-01-25_r6
echo " "
python main.py random --parameters parameters/reward_function_r7.json --base_path results/random/2022-01-26_r7
echo " "
python main.py ac --parameters parameters/reward_function_r5.json --base_path results/actor_critic/2022-01-27_r5
echo " "
python main.py ac --parameters parameters/reward_function_r5_masked.json --base_path results/actor_critic/2022-01-28_r5_masked
echo " "
python main.py ac --parameters parameters/reward_function_r6.json --base_path results/actor_critic/2022-01-29_r6
echo " "
python main.py ac --parameters parameters/reward_function_r7_masked.json --base_path results/actor_critic/2022-01-30_r7_masked
echo " "