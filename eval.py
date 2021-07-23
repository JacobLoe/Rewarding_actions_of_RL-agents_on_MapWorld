from utils import create_figure, create_histogram, create_all_plots, get_data
import os
import argparse
import glob


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default=None,
                        help="Path to the raw results files. Use this if plots for only one set of results are wanted."
                             "The path is assumed to be in the form 'results/**/**/raw'")
    parser.add_argument('--save_plots', type=bool, default=True,
                        help='Use --save_plots '' to show the plots in the browser')
    parser.add_argument('--filter_return', type=bool, default=True,
                        help='Sets whether to apply a moving average to the return of the model')
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