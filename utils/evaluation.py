from tqdm import tqdm
import matplotlib.pyplot as plt


def eval_rand_baseline(mapgame, rand_baseline):
    """

    Args:
        mapgame:
        rand_baseline:

    Returns:

    """
    _ = mapgame.reset()
    available_actions = mapgame.total_available_actions
    while not mapgame.done:
        i = rand_baseline.select_action(len(available_actions))
        action = available_actions[i]
        _ = mapgame.step(action)
    return mapgame.model_return, mapgame.model_steps


def eval_rl_baseline(mapgame, rl_baseline):
    """

    Args:
        mapgame:
        rl_baseline:

    Returns:

    """
    return 0


def eval_hrl_model(mapgame, hrl_baseline):
    """

    Args:
        mapgame:
        hrl_baseline:

    Returns:

    """
    return 0


def evaluate_model(mapgame, model, eval_function, num_iterations=10):
    """

    Args:
        mapgame:
        model:
        eval_function:
        num_iterations:

    Returns:

    """
    model_return = []
    model_steps = []
    for i in tqdm(range(num_iterations)):

        r, s = eval_function(mapgame, model)
        model_return.append(r)
        model_steps.append(s)

    return model_return, model_steps


def create_histograms(total_return, steps):
    """

    Args:
        total_return:
        steps:
    """
    # TODO FIX width of boxes, fix weird bins
    r_bins = range(int(min(total_return)), int(max(total_return))+10, 10)
    s_bins = range(int(min(steps)), int(max(steps))+1)

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(total_return, bins=r_bins)
    axs[0].set_ylabel('Count')
    axs[0].set_xlabel('Return')
    axs[1].hist(steps, bins=s_bins)
    axs[1].set_ylabel('Count')
    axs[1].set_xlabel('Total steps')
    plt.show()
