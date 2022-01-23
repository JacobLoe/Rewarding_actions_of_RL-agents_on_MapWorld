#!/bin/bash
#python main.py random --parameters parameters/reward_function_r1.json --base_path results/random/2022-01-12_r1
#echo " "
python main.py random --parameters parameters/reward_function_r3.json --base_path results/random/2022-01-14_r3
echo " "
python main.py random --parameters parameters/reward_function_r5.json --base_path results/random/2022-01-16_r5
echo " "
python main.py random --parameters parameters/reward_function_r6.json --base_path results/random/2022-01-18_r6
echo " "
python main.py random --parameters parameters/reward_function_r7.json --base_path results/random/2022-01-20_r7
echo " "
python main.py random --parameters parameters/reward_function_r8.json --base_path results/random/2022-01-22_r8
echo " "
python main.py random --parameters parameters/reward_function_r3_masked.json --base_path results/random/2022-01-24_r3_masked
echo " "
python main.py random --parameters parameters/reward_function_r5_masked.json --base_path results/random/2022-01-26_r5_masked
echo " "
python main.py random --parameters parameters/reward_function_r6_masked.json --base_path results/random/2022-01-28_r6_masked
echo " "
python main.py random --parameters parameters/reward_function_r8_masked.json --base_path results/random/2022-01-30_r8_masked
