# Rewarding action of RL-agents on MapWorld

Trains and evaluates a reinforcement learning agent in moving through a graph, conditioned only on textual descriptions and images at each node on the grpah. For further information refer to the `IM_report.pdf`.

Create and activate conda environment 
(Tested with Anaconda version 4.10.1):
```
conda env create -f environment.yml
conda activate mapworld
```

Download the ADE20K dataset and extract its content into the project folder,
The directory structure of the dataset is shown below:
```
ADE20K_2021_17_01
├── images
│   ├── ADE
│   │   ├── training
│   │   ├── validation
```
The used version of ADE20K is 2021_17_01. 

Before running the main script two scripts have to be run first:

```
python localized_narratives/localized_narratives.py
python utils/distances.py
```

An experiment with an actor-critic (or random) agent and reward function r_5 can be started with the following commands:
```
python main.py ac --parameters parameters/reward_function_r5.json --base_path results/actor-crtic/test_r5

python main.py random --parameters parameters/reward_function_r5.json --base_path results/random/test_r5
```
