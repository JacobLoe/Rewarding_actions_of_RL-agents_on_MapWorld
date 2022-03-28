# Rewarding action of RL-agents on MapWorld
Create and activate conda environment 
(Tested with Anaconda version 4.10.1):
```
conda env create -f environment.yml
conda activate mapworld
```

Download the ADE20K dataset and extract its content into the project folder,
The used version of ADE20K is 2021_17_01.

Before running the main script two scripts have to be run first:

```
python localized_narratives/localized_narratives.py
python utils/distances.py
```
