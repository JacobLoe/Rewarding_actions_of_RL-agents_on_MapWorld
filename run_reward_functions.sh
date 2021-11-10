#!/bin/bash

python main.py ac --parameters params/reward_funtion_r4.json --base_path results/actor_critic/2021-11-10_reward_rf4
echo " "
python main.py ac --parameters params/reward_funtion_r5.json --base_path results/actor_critic/2021-11-12_reward_rf5
echo " "
python main.py ac --parameters params/reward_funtion_r6.json --base_path results/actor_critic/2021-11-14_reward_rf6
echo " "
python main.py ac --parameters params/reward_funtion_r7.json --base_path results/actor_critic/2021-11-16_reward_rf7
echo " "
python main.py ac --parameters params/reward_funtion_r8.json --base_path results/actor_critic/2021-11-18_reward_rf8
echo " "
python main.py ac --parameters params/reward_funtion_r9.json --base_path results/actor_critic/2021-11-20_reward_rf9
