from agents import RandomBaseline
from MapWorld import MapWorldGym
from utils import evaluation
import numpy as np

if __name__ == '__main__':
    mwg = MapWorldGym()
    # initial_state = mwg.reset()
    # print(np.shape(initial_state))
    # print(initial_state)
    # print(mwg.target_room)
    # directions = mwg.directions
    # num_actions = mwg.total_num_actions
    # print(num_actions, directions)

    rb = RandomBaseline()

    model_return, model_steps = evaluation.evaluate_model(mwg, rb, evaluation.eval_rand_baseline, num_iterations=5)
    print('\n-------------------')
    print('Return per model run: ', model_return)
    print('Mean return: ', np.mean(model_return))
    print('-------------------')
    print('Total steps per model run', model_steps)
    print('Mean steps: ', np.mean(model_steps))

    evaluation.create_histograms(model_return, model_steps)
