from agents import run_random_baseline, reinforce
from MapWorld import MapWorldGym
from utils import save_parameters, save_results, create_figure, create_histogram
import numpy as np
import time
import json
import os
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=['random', 'rl', 'eval'], help="")
    parser.add_argument("--base_path", default="results", help="")
    parser.add_argument("--save", default="True", help="")
    parser.add_argument("--parameters", default='all_parameters.json', help="")
    args = parser.parse_args()

    if args.model == 'random' or args.model == 'rl':
        with open(args.parameters, 'r') as fp:
            parameters = json.load(fp)

        mw_params = parameters['MapWorld']
        mwg = MapWorldGym(n=mw_params['n'], m=mw_params['m'], n_rooms=mw_params['n_rooms'],
                          room_types=mw_params['room_types'], room_repetitions=mw_params['room_repetitions'],
                          ade_path=mw_params['ade_path'],
                          caption_checkpoints=mw_params['caption_checkpoints'],
                          caption_vocab=mw_params['caption_vocab'],
                          image_resolution=(mw_params['image_width'], mw_params['image_height']))

    if args.model == 'random':
        model_return, model_steps, model_hits = run_random_baseline(mwg,
                                                                    episodes=parameters['training']['num_episodes'])
        save_results(model_return, model_steps, model_hits, args.base_path)

    elif args.model == 'rl':
        # save parameters before running the model
        parameters = {'rl_baseline': parameters['rl_baseline'],
                      'training': parameters['training'],
                      'MapWorld': mw_params}
        save_parameters(parameters, args.base_path)
        model_return, model_steps, model_hits = reinforce(mwg,
                                                          parameters['rl_baseline'],
                                                          parameters['training'],
                                                          base_path=args.base_path)
        save_results(model_return, model_steps, model_hits, args.base_path)

    elif args.model == 'eval':
        model_return = np.load(os.path.join(args.base_path, 'model_return.npy'))
        model_hits = np.load(os.path.join(args.base_path, 'model_hits.npy'))
        model_steps = np.load(os.path.join(args.base_path, 'model_steps.npy'))

    # model_steps = np.cumsum(model_steps)
    print('\n-------------------')
    # print('Return per model run: ', model_return)
    print('Mean return: ', np.mean(model_return))
    print('-------------------')
    # print('Total steps per model run', model_steps)
    # print('Cumulative steps', np.cumsum(model_steps))
    print('Mean steps: ', np.mean(model_steps))
    print('-------------------')
    # print('model_hits', model_hits)
    print('accuracy', np.sum(model_hits)/len(model_hits))
    print('-------------------')
    print('Episodes: ', len(model_return))

    # create_histogram(model_return, 'return')
    # create_figure(model_steps, model_return, 'REINFORCE', args.base_path)
