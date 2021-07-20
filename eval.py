from utils import create_figure, create_histogram
import numpy as np
import os
import argparse
import json
import glob


def get_data(base_path):
    """
    Loads the data. Assumes base_path is structured like: "results/**/**/raw"
    Args:
        base_path: string, path to folder where the data is saved to
    Returns: Results (return, steps, hits) as numpy arrays,
    """
    model_return = np.load(os.path.join(base_path, 'model_return.npy'))
    model_hits = np.load(os.path.join(base_path, 'model_hits.npy'))
    model_steps = np.load(os.path.join(base_path, 'model_steps.npy'))

    with open(os.path.join(base_path, 'model_parameters.json'), 'r') as fp:
        parameters = json.load(fp)
    num_episodes = parameters['training']['num_episodes']

    plot_base_path = os.path.join(os.path.split(base_path)[0], 'plots')

    return model_return, model_steps, model_hits, num_episodes, plot_base_path


def create_all_plots(model_return, model_steps, model_hits, num_episodes,
                     plot_base_path, save_plots, filter_return, filter_size):
    """

    Args:
        model_return:
        model_steps:
        model_hits:
        num_episodes:
        plot_base_path:
        save_plots:
        filter_size:
    """
    title = 'the return over {}'.format(num_episodes)
    plot_path = os.path.join(plot_base_path, 'return_histogram.png')
    create_histogram(model_return, title, plot_path, save_plot=save_plots)

    title = 'room guesses over {}'.format(num_episodes)
    plot_path = os.path.join(plot_base_path, 'hits_histogram.png')
    create_histogram(model_hits, title, plot_path, save_plot=save_plots)

    title = 'the steps over {}'.format(num_episodes)
    plot_path = os.path.join(plot_base_path, 'steps_histogram.png')
    create_histogram(model_steps, title, plot_path, save_plot=save_plots)

    plot_path = os.path.join(plot_base_path, 'return_over_episodes.png')
    create_figure(model_steps, model_return, 'REINFORCE', plot_path, save_plot=save_plots,
                  filter_return=filter_return, size=filter_size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default=None,
                        help="Path to the raw results files. Use this if plots for only one set of results are wanted."
                             "The path is assumed to be in the form 'results/**/**/raw'")
    parser.add_argument('--save_plots', type=bool, default=True,
                        help='Use --save_plots '' to show the plots in the browser')
    parser.add_argument('--filter_return', type=bool, default=True,
                        help='Sets whether to apply a moving average to the return')
    parser.add_argument('--filter_size', type=int, default=100, help='Sets the size of the moving average filter')
    args = parser.parse_args()

    if args.base_path:
        print(f'Creating plots {args.base_path}')

        model_return, model_steps, model_hits, num_episodes, plot_base_path = get_data(args.base_path)

        if not os.path.isdir(plot_base_path):
            os.makedirs(plot_base_path)

        create_all_plots(model_return, model_steps, model_hits, num_episodes, plot_base_path,
                         args.save_plots, args.filter_return, args.filter_size)
    else:
        results_paths = glob.glob('results/**/*.json', recursive=True)

        for rp in results_paths:
            base_path = os.path.split(rp)[0]
            print(f'Creating plots {base_path}')
            model_return, model_steps, model_hits, num_episodes, plot_base_path = get_data(base_path)

            if not os.path.isdir(plot_base_path):
                os.makedirs(plot_base_path)

            create_all_plots(model_return, model_steps, model_hits, num_episodes, plot_base_path,
                             args.save_plots, args.filter_return, args.filter_size)