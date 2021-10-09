import numpy as np
import plotly.express as px
import pandas as pd
from scipy.ndimage.filters import uniform_filter1d
import os
import json


def get_data(base_path):
    """
    Loads the data. Assumes base_path is structured like: "results/**/**/raw"
    Args:
        base_path: string, path to folder where the data is saved to
    Returns: Results (return, steps, hits) as numpy arrays,
    """
    model_return = np.load(os.path.join(base_path, 'model_return.npy'))
    model_hits = np.load(os.path.join(base_path, 'model_hits.npy'))
    model_steps = np.load(os.path.join(base_path, 'model_steps.npy'))

    with open(os.path.join(base_path, 'model_parameters.json'), 'r') as fp:
        parameters = json.load(fp)
    num_episodes = parameters['training']['num_episodes']

    model_name = [n for n in parameters.keys() if n != 'MapWorld' and n != 'training'][0]

    plot_base_path = os.path.join(base_path, 'plots')

    return model_return, model_steps, model_hits, num_episodes, plot_base_path, model_name


def plot_accuracy(model_hits, split=100, plot_path='', save_plot=True, save_html=False):
    accuracy = np.sum(model_hits)/(len(model_hits))

    split_model_hits = np.array_split(model_hits, split)
    split_size = len(split_model_hits[0])
    accuracy_per_split = [np.sum(x)/len(x) for x in split_model_hits]

    title = f'Accuracy per length {split_size} chunks. Total accuracy: {accuracy}'
    x_axis_label = 'chunk'
    y_axis_label = 'accuracy per chunk'

    fig = px.line(x=range(len(split_model_hits)),
                  y=accuracy_per_split,
                  title=title)
    fig.update_xaxes(title_text=x_axis_label)
    fig.update_yaxes(title_text=y_axis_label)
    if save_plot:
        fig.write_image(plot_path)
        if save_html:
            html_path = plot_path[:-4] + '.html'
            fig.write_html(html_path)
    else:
        fig.show()


def create_histogram(data, title, plot_path='', save_plot=True, save_html=False):
    """

    Args:
        save_html:
        data:
        title:
        plot_path:
        save_plot:
    """
    # TODO better title
    title = f'Counts of {title}'

    df = pd.DataFrame(data)
    fig = px.histogram(df, title=title)
    if save_plot:
        fig.write_image(plot_path)
        if save_html:
            html_path = plot_path[:-4] + '.html'
            fig.write_html(html_path)
    else:
        fig.show()


def return_over_episodes(model_steps, model_return, model_name, plot_path,
                         save_plot=True, filter_return=True, size=100, save_html=False):
    """

    Args:
        save_html:
        filter_return:
        size:
        save_plot:
        model_steps:
        model_return:
        model_name:
        plot_path:
    """

    if filter_return:
        mreturn = uniform_filter1d(model_return, mode='reflect', size=size)
    else:
        mreturn = model_return

    title = f'Return of {model_name} for {len(model_return)} episodes, moving average over {size} episodes'
    x_axis_label = 'Episodes'
    y_axis_label = 'Return'
    fig = px.line(x=np.cumsum(model_steps),
                  y=mreturn,
                  title=title,
                  )
    fig.update_xaxes(title_text=x_axis_label)
    fig.update_yaxes(title_text=y_axis_label)
    if save_plot:
        fig.write_image(plot_path)
        if save_html:
            html_path = plot_path[:-4] + '.html'
            fig.write_html(html_path)
    else:
        fig.show()


def steps_over_episodes(model_steps, model_name, plot_path,
                        save_plot=True, save_html=False):
    """

    Args:
        save_html:
        save_plot:
        model_steps:
        model_name:
        plot_path:
    """

    title = f'Steps of {model_name} for every episode'
    x_axis_label = 'episode'
    y_axis_label = 'Steps per episode'
    fig = px.line(x=np.cumsum(model_steps),
                  y=model_steps,
                  title=title,
                  )
    fig.update_xaxes(title_text=x_axis_label)
    fig.update_yaxes(title_text=y_axis_label)
    if save_plot:
        fig.write_image(plot_path)
        if save_html:
            html_path = plot_path[:-4] + '.html'
            fig.write_html(html_path)
    else:
        fig.show()


def create_all_plots(model_name, model_return, model_steps, model_hits, num_episodes,
                     plot_base_path, save_plots, filter_return, filter_size, save_html, split):
    """

    Args:
        model_name:
        save_html:
        model_return:
        model_steps:
        model_hits:
        num_episodes:
        plot_base_path:
        save_plots:
        filter_return: bool, Sets whether to apply a moving average to the return of the model
        filter_size:
    """
    title = f'the return over {num_episodes}'
    plot_path = os.path.join(plot_base_path, 'return_histogram.png')
    create_histogram(model_return, title, plot_path, save_plot=save_plots, save_html=save_html)

    title = f'room guesses over {num_episodes}'
    plot_path = os.path.join(plot_base_path, 'hits_histogram.png')
    create_histogram(model_hits, title, plot_path, save_plot=save_plots, save_html=save_html)

    title = f'the steps over {num_episodes}'
    plot_path = os.path.join(plot_base_path, 'steps_histogram.png')
    create_histogram(model_steps, title, plot_path, save_plot=save_plots, save_html=save_html)

    plot_path = os.path.join(plot_base_path, 'return_over_episodes.png')
    return_over_episodes(model_steps, model_return, model_name, plot_path, save_plot=save_plots,
                         filter_return=filter_return, size=filter_size, save_html=save_html)

    plot_path = os.path.join(plot_base_path, 'steps_over_episodes.png')
    steps_over_episodes(model_steps, model_name, plot_path, save_plot=save_plots, save_html=save_html)

    plot_path = os.path.join(plot_base_path, 'accuracy.png')
    plot_accuracy(model_hits, split, plot_path, save_plot=save_plots, save_html=save_html)
