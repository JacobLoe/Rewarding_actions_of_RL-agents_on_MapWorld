from agents import RandomBaseline
from MapWorld import MapWorldGym
import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
from utils.evaluation import eval_rand_baseline, evaluate_model


if __name__ == '__main__':
    mwg = MapWorldGym()
    # initial_state = mwg.reset()
    # print(np.shape(initial_state))
    # print(initial_state)
    # print(mwg.target_room)
    # available_actions = mwg.available_actions
    # num_actions = mwg.num_actions
    # print(num_actions, available_actions)
    rb = RandomBaseline()

    model_return, model_steps = evaluate_model(mwg, rb, eval_rand_baseline)
    print(model_return)
    print(np.mean(model_return))
    print('-------------------')
    print(model_steps)
    print(np.mean(model_steps))
