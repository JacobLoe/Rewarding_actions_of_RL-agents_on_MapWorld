import numpy as np
from tqdm import tqdm


def run_random_baseline(mapgame, episodes=20000):
    """

    Args:
        mapgame:
        episodes:
    Returns:

    """
    model_return = []
    model_steps = []
    hits = []
    for _ in tqdm(range(episodes)):

        _ = mapgame.reset()
        available_actions = mapgame.total_available_actions
        done = False
        while not done:
            action = np.random.randint(0, len(available_actions))
            _, _, done, room_found = mapgame.step(action)

        model_return.append(mapgame.model_return)
        model_steps.append(mapgame.model_steps)
        hits.append(room_found)

    return model_return, model_steps, hits
