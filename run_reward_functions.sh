#!/bin/bash
python main.py ac --parameters parameters/reward_function_r1_masked.json --base_path results/actor_critic/2022-04-27_r1_masked_1
echo " "
python main.py ac --parameters parameters/reward_function_r1_masked.json --base_path results/actor_critic/2022-04-29_r1_masked_2
echo " "
python main.py ac --parameters parameters/reward_function_r1_masked.json --base_path results/actor_critic/2022-04-30_r1_masked_3
echo " "