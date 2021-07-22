from agents import run_random_baseline, reinforce
from MapWorld import MapWorldGym
from utils import save_parameters, save_results
import numpy as np
import json
import os
import argparse
import logging

logger = logging.getLogger(__name__)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False    # prevent log messages from appearing twice


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=['random', 'rl'], help="")
    parser.add_argument("--base_path", default="results", help="")
    parser.add_argument("--parameters", default='all_parameters.json'
                        , help="The path to the ")
    parser.add_argument('--log_level', default='warning', choices=['warning', 'info', 'debug'],
                        help='set which logging messages to print')
    parser.add_argument('--save_results', type=bool, default=True, help='')
    args = parser.parse_args()

    # set log level according to command line
    log_level = {'warning': logging.WARNING, 'info': logging.INFO, 'debug': logging.DEBUG}
    logging.basicConfig(level=log_level[args.log_level])

    with open(args.parameters, 'r') as fp:
        parameters = json.load(fp)

    mw_params = parameters['MapWorld']
    mwg = MapWorldGym(n=mw_params['n'], m=mw_params['m'], n_rooms=mw_params['n_rooms'],
                      room_types=mw_params['room_types'], room_repetitions=mw_params['room_repetitions'],
                      ade_path=mw_params['ade_path'],
                      image_resolution=(mw_params['image_width'], mw_params['image_height']),
                      captions=mw_params['captions'])
    if args.model == 'random':
        parameters = {'training': parameters['training'],
                      'MapWorld': mw_params}
        if args.save_results:
            save_parameters(parameters, args.base_path)
        model_return, model_steps, model_hits = run_random_baseline(mwg, logger,
                                                                    episodes=parameters['training']['num_episodes'])
        if args.save_results:
            save_results(model_return, model_steps, model_hits, args.base_path)

    elif args.model == 'rl':
        # save parameters before running the model
        parameters = {'rl_baseline': parameters['rl_baseline'],
                      'training': parameters['training'],
                      'MapWorld': mw_params}
        if args.save_results:
            save_parameters(parameters, args.base_path)
        model_return, model_steps, model_hits = reinforce(mwg,
                                                          parameters['rl_baseline'],
                                                          parameters['training'],
                                                          base_path=args.base_path,
                                                          logger=logger)
        if args.save_results:
            save_results(model_return, model_steps, model_hits, args.base_path)

    print('\n-------------------')
    print('Mean return: ', np.mean(model_return))
    print('-------------------')
    print('Mean steps: ', np.mean(model_steps))
    print('-------------------')
    print('accuracy', np.sum(model_hits)/len(model_hits))
    print('-------------------')
    print('Episodes: ', len(model_return))
