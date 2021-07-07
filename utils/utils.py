from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import date

def save_parameters_and_results(parameters, model_return, model_steps, model_hits):
    """
    Saves the results and parameters for a training run of a model,
    The files are saved under "results/raw" in a folder specific to the training run.
    The folder is named "YYYY-MM-DD_NUM_EPISODES'
    Results are saved as numpy files, parameters as json.
    Args:
        parameters: dict,
        model_return: list,
        model_steps: list
        model_hits: list
    """
    f = str(date.today()) + '_' + str(parameters['training']['num_episodes'])
    base_path = os.path.join('results/raw', f)
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    json_path = os.path.join(base_path, 'model_parameters.json')

    with open(json_path, 'w') as fp:
        json.dump(parameters, fp, sort_keys=True, indent=4)

    np.save(os.path.join(base_path, 'model_steps'), model_steps)
    np.save(os.path.join(base_path, 'model_return'), model_return)
    np.save(os.path.join(base_path, 'model_hits'), model_hits)


def create_histograms(total_return, steps):
    """

    Args:
        total_return:
        steps:
    """
    # TODO FIX width of boxes
    # TODO fix weird bins
    r_bins = range(int(min(total_return)), int(max(total_return))+10, 10)
    s_bins = range(int(min(steps)), int(max(steps))+1)

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(total_return, bins=r_bins)
    axs[0].set_ylabel('Count')
    axs[0].set_xlabel('Return')
    axs[1].hist(steps, bins=s_bins)
    axs[1].set_ylabel('Count')
    axs[1].set_xlabel('Total steps')
    plt.show()
