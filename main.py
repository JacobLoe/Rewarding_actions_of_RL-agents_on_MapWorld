from agents import RandomBaseline
from agents.rl_baseline import reinforce
from MapWorld import MapWorldGym
from utils import evaluation
import numpy as np
import time
from tqdm import tqdm


if __name__ == '__main__':
    mwg = MapWorldGym()

    # reinforce(mwg)
