import numpy as np
import os
import json


def save_results(model_return, model_steps, model_hits, base_path):
    """
    Saves the results for a training run of a model,
    Results are saved as numpy files.
    Args:

        model_return: list,
        model_steps: list
        model_hits: list
        base_path: string, base path where the results are saved to
    """
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    np.save(os.path.join(base_path, 'model_steps'), model_steps)
    np.save(os.path.join(base_path, 'model_return'), model_return)
    np.save(os.path.join(base_path, 'model_hits'), model_hits)


def save_parameters(parameters, base_path):
    """
    Saves the parameters for a training run of a model,
    Parameters are saved as a json file.
    Args:
        parameters: dict,
        base_path: string, base path where the results are saved to
    """
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    json_path = os.path.join(base_path, 'model_parameters.json')

    with open(json_path, 'w') as fp:
        json.dump(parameters, fp, sort_keys=True, indent=4)


def preprocess_mapworld_state(state, em_model):
    im = state['current_room']
    im = np.reshape(im, (np.shape(im)[2], np.shape(im)[1], np.shape(im)[0]))

    text = state['text_state']
    embeddings = em_model.encode(text)
    return im, embeddings
