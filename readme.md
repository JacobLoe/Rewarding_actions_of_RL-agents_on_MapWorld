# HRL for text based games
How to create a conda environment for hierarchical reinforcement learning.

(Tested with Anaconda version 4.9.2)

Create and activate conda environment:
```
conda env create -f environment.yml
conda activate hrl-tb
```

Download weights for im2txt from https://www.dropbox.com/s/87dm6ly33845p72/im2txt_5M.zip?dl=0 and extract the content to im2txt/checkpoints

For using jupyter notebooks run this in the conda environment:
```
python -m ipykernel install --user --name=hrl-tb
```