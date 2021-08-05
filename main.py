from agents import random_baseline, reinforce, actor_critic
from agents.dqn import main
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
logger.propagate = False    # prevents log messages from appearing twice


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=['random', 'reinforce', 'ac', 'dqn'], help="")
    parser.add_argument("--base_path", default="results",
                        help="Path where results, checkpoints and parameters are saved to")
    parser.add_argument("--parameters", default='all_parameters.json'
                        , help="The path to the global parameters json")
    parser.add_argument('--log_level', default='warning', choices=['warning', 'info', 'debug'],
                        help='Sets which logging messages to print')
    parser.add_argument('--save_results', type=bool, default=True,
                        help='S')
    parser.add_argument('--load_checkpoint', type=bool, default='',
                        help='If set to True, parameters and checkpoints are loaded from args.base_path')
    args = parser.parse_args()

    # set log level according to command line
    log_level = {'warning': logging.WARNING, 'info': logging.INFO, 'debug': logging.DEBUG}
    logging.basicConfig(level=log_level[args.log_level])

    # load old parameters if a model is run from a checkpoint
    if not args.load_checkpoint:
        with open(args.parameters, 'r') as fp:
            parameters = json.load(fp)
    else:
        param_path = os.path.join(args.base_path, 'model_parameters.json')
        with open(param_path, 'r') as fp:
            parameters = json.load(fp)

    # initialise a MapWorld-object from the parameters
    mw_params = parameters['MapWorld']
    mwg = MapWorldGym(n=mw_params['n'], m=mw_params['m'], n_rooms=mw_params['n_rooms'],
                      room_types=mw_params['room_types'], room_repetitions=mw_params['room_repetitions'],
                      ade_path=mw_params['ade_path'],
                      image_resolution=(mw_params['image_width'], mw_params['image_height']),
                      captions=mw_params['captions'])

    # run the chosen model on MapWorld with the loaded parameters
    if args.model == 'random':
        parameters = {'training': parameters['training'],
                      'MapWorld': mw_params}
        if args.save_results:
            save_parameters(parameters, args.base_path)
        model_return, model_steps, model_hits = random_baseline(mwg, logger,
                                                                episodes=parameters['training']['num_episodes'],
                                                                max_steps=parameters['training']['num_episodes'])
        if args.save_results:
            save_results(model_return, model_steps, model_hits, args.base_path)

    elif args.model == 'reinforce':
        # save parameters before running the model
        parameters = {'REINFORCE': parameters['REINFORCE'],
                      'training': parameters['training'],
                      'MapWorld': mw_params}
        if args.save_results:
            save_parameters(parameters, args.base_path)
        model_return, model_steps, model_hits = reinforce(mwg,
                                                          parameters['REINFORCE'],
                                                          parameters['training'],
                                                          base_path=args.base_path,
                                                          logger=logger,
                                                          save_results=args.save_results)
        if args.save_results:
            save_results(model_return, model_steps, model_hits, args.base_path)

    elif args.model == 'ac':
        parameters = {'actor_critic': parameters['actor_critic'],
                      'training': parameters['training'],
                      'MapWorld': mw_params}
        model_return, model_steps, model_hits = actor_critic(mwg,
                                                             parameters['actor_critic'],
                                                             parameters['training'],
                                                             base_path=args.base_path,
                                                             logger=logger,
                                                             save_results=args.save_results)
    elif args.model == 'dqn':
        parameters = {'dqn': parameters['dqn'],
                      'training': parameters['training'],
                      'MapWorld': mw_params}
        main(mwg,
             parameters['dqn'],
             parameters['training'])
    print('\n-------------------')
    print('Mean return: ', np.mean(model_return))
    print('-------------------')
    print('Mean steps: ', np.mean(model_steps))
    print('-------------------')
    print('accuracy', np.sum(model_hits)/len(model_hits))
    print('-------------------')
    print('Episodes: ', len(model_return))
