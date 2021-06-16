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
    for i in range(num_iterations):

        r, s = eval_function(mapgame, model)
        model_return.append(r)
        model_steps.append(s)

    return model_return, model_steps
