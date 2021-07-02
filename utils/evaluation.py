from tqdm import tqdm
import matplotlib.pyplot as plt


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


def create_histograms(total_return, steps):
    """

    Args:
        total_return:
        steps:
    """
    # TODO FIX width of boxes
    # TODO fix weird bins
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
