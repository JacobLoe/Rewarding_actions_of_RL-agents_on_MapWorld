from utils import create_all_plots, get_data
import os
import argparse
import glob


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default='results',
                        help="Path to the raw results files. Use this if plots for only one set of results are wanted.")
    parser.add_argument('--save_plots', type=bool, default=True,
                        help='Use --save_plots '' to show the plots in the browser')
    parser.add_argument('--save_html', type=bool, default='',
                        help='Use --save_html '' to save the plots as html in addition to the .png-file')
    parser.add_argument('--filter_return', type=bool, default=True,
                        help='Sets whether to apply a moving average to the return of the model')
    parser.add_argument('--filter_size', type=int, default=50000, help='Sets the size of the moving average filter')
    parser.add_argument('--split', type=int, default=100, help='Defines how often the accuracy is calculated ...')
    args = parser.parse_args()

    print(f'Collecting parameter json-files in directory: "{args.base_path}"\n')
    results_paths = os.path.join(args.base_path, '**/model_parameters.json')
    parameter_jsons = glob.glob(results_paths, recursive=True)

    for rp in parameter_jsons:
        base_path = os.path.split(rp)[0]
        data_dataframe, num_episodes, plot_base_path, model_name = get_data(base_path)
        print(f'Creating plots for {model_name}')
        if not os.path.isdir(plot_base_path):
            os.makedirs(plot_base_path)

        create_all_plots(model_name, data_dataframe, plot_base_path,
                         args.save_plots, args.filter_return, args.filter_size, args.save_html, args.split)
        print('\n')
