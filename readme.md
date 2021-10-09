# HRL for text based games
How to create a conda environment for hierarchical reinforcement learning.

(Tested with Anaconda version 4.10.1)

Create and activate conda environment:
```
conda env create -f environment.yml
conda activate hrl-tb
```

Before running the code two scripts have to be run first:

```
python localized_narratives/localized_narratives.py
python utils/dists.py
```
