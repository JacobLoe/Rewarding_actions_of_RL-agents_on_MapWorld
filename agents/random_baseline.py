import numpy as np
from tqdm import tqdm
from time import time


def random_baseline(mapgame, episodes=200000, max_steps=50):
    """

    Args:
        max_steps:
        mapgame:
        episodes:
    Returns:

    """
    model_return = []
    model_steps = []
    hits = []
    for _ in tqdm(range(int(episodes))):

        t_r = time()
        s = mapgame.reset()

        # print(f'Time for env reset {time() - t_r}')
        available_actions = mapgame.actions
        done = False
        steps = 0
        while not done and steps < max_steps:
            t_s = time()
            action = np.random.randint(0, len(available_actions))
            s, _, done, room_found = mapgame.step(action)
            # print(f'Time for env step {time()-t_s}')
            steps += 1
        t_a = time()
        model_return.append(mapgame.model_return)
        model_steps.append(mapgame.model_steps)
        hits.append(room_found)
        # print(f'Time for append {time()-t_a}')
        # print(f'Time for an episode {time()-t_r} \n')

    return model_return, model_steps, hits
