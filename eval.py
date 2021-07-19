from utils import create_figure, create_histogram
import numpy as np
import os
import argparse
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", help="Path to the raw results files")
    parser.add_argument('--save_plots', type=bool, default=True, help='')
    args = parser.parse_args()

    model_return = np.load(os.path.join(args.base_path, 'model_return.npy'))
    model_hits = np.load(os.path.join(args.base_path, 'model_hits.npy'))
    model_steps = np.load(os.path.join(args.base_path, 'model_steps.npy'))
    with open(os.path.join(args.base_path, 'model_parameters.json'), 'r') as fp:
        parameters = json.load(fp)

    plot_base_path = os.path.join(os.path.split(args.base_path)[0], 'plots')
    if not os.path.isdir(plot_base_path):
        os.makedirs(plot_base_path)

    num_episodes = parameters['training']['num_episodes']

    title = 'the return over {}'.format(num_episodes)
    plot_path = os.path.join(plot_base_path, 'return_histogram.png')
    create_histogram(model_return, title, plot_path, save_plot=args.save_plots)

    title = 'room guesses over {}'.format(num_episodes)
    plot_path = os.path.join(plot_base_path, 'hits_histogram.png')
    create_histogram(model_hits, title, plot_path, save_plot=args.save_plots)

    plot_path = os.path.join(plot_base_path, 'return_over_episodes.png')
    create_figure(model_steps, model_return, 'REINFORCE', plot_path, save_plot=args.save_plots, size=100)
