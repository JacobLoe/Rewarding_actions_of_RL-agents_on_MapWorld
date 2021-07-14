import matplotlib.pyplot as plt
import numpy as np
import os
import json
import plotly.express as px


def save_results(model_return, model_steps, model_hits, base_path):
    """
    Saves the results for a training run of a model,
    Results are saved as numpy files.
    Args:

        model_return: list,
        model_steps: list
        model_hits: list
        base_path: string, base path where the results are saved to
    """
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    np.save(os.path.join(base_path, 'model_steps'), model_steps)
    np.save(os.path.join(base_path, 'model_return'), model_return)
    np.save(os.path.join(base_path, 'model_hits'), model_hits)


def save_parameters(parameters, base_path):
    """
    Saves the parameters for a training run of a model,
    Parameters are saved as a json file.
    Args:
        parameters: dict,
        base_path: string, base path where the results are saved to
    """
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    json_path = os.path.join(base_path, 'model_parameters.json')

    with open(json_path, 'w') as fp:
        json.dump(parameters, fp, sort_keys=True, indent=4)


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


def create_figure(model_steps, model_return, model_name, base_path):
    title = 'Return of {} over {} episodes'.format(model_name, len(model_return))
    x_axis_label = 'Steps'
    y_axis_label = 'Return'
    fig = px.line(x=np.cumsum(model_steps),
                  y=model_return,
                  title=title,
                  )
    fig.update_xaxes(title_text=x_axis_label)
    fig.update_yaxes(title_text=y_axis_label)
    fig.write_image(os.path.join(base_path, 'fig.png'))