from agents import run_random_baseline, reinforce
from MapWorld import MapWorldGym
from utils import evaluation
import numpy as np
import time
import plotly.express as px

if __name__ == '__main__':
    mwg = MapWorldGym()
    #
    # model_return, model_steps, hits = run_random_baseline(mwg, num_iterations=5)

    # model_return = [-400.0, -500.0, -600.0, -300.0, -600.0]
    # model_steps = [4, 9, 15, 23, 29]
    # hits = [0, 1, 0, 1, 1]
    #
    # np.save('results/raw/model_steps', model_steps)
    # np.save('results/raw/model_return', model_return)
    # np.save('results/raw/hits', hits)
    #
    # model_steps = np.cumsum(model_steps)
    # print('\n-------------------')
    # print('Return per model run: ', model_return)
    # print('Mean return: ', np.mean(model_return))
    # print('-------------------')
    # print('Total steps per model run', model_steps)
    # print('Cumulative steps', np.cumsum(model_steps))
    # print('Mean steps: ', np.mean(model_steps))
    #
    # fig = px.line(x=np.cumsum(model_steps),
    #               y=model_return,
    #               title='Return over 5 episodes',
    #               )
    # fig.update_xaxes(title_text='Steps')
    # fig.update_yaxes(title_text='Return')
    # fig.show()

    model_return, model_steps, hits = reinforce(mwg)
