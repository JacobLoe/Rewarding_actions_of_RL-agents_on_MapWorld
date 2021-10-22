import numpy as np
import plotly.express as px
import pandas as pd
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

    model_name = [n for n in parameters.keys() if n != 'MapWorld' and n != 'training'][0] + os.path.split(base_path)[1][10:]

    plot_base_path = os.path.join(base_path, 'plots')

    data_dict = {'model_return': model_return, 'model_steps': model_steps, 'model_hits': model_hits}
    data_dataframe = pd.DataFrame(data_dict)
    return data_dataframe, num_episodes, plot_base_path, model_name


def plot_accuracy(model_hits, model_name, plot_path, split=100, save_plot=True, save_html=False):
    # compute total accuracy
    accuracy = np.sum(model_hits)/(len(model_hits))

    # create splits of even length of the data
    split_model_hits = np.array_split(model_hits, split)
    split_size = len(split_model_hits[0])
    # compute accuracy over
    accuracy_per_split = [np.sum(x)/len(x) for x in split_model_hits]

    title = f'Accuracy of {model_name} per part of episode length {split_size}. Total accuracy: {accuracy}'
    x_axis_label = 'Part'
    y_axis_label = 'Accuracy'

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


def create_histogram(data_dataframe, title, plot_path='', save_plot=True, save_html=False):
    """

    Args:
        save_html:
        data_dataframe:
        title:
        plot_path:
        save_plot:
    """

    # TODO where are the axis descriptions ?
    fig = px.histogram(data_dataframe, title=title)
    if save_plot:
        fig.write_image(plot_path)
        if save_html:
            html_path = plot_path[:-4] + '.html'
            fig.write_html(html_path)
    else:
        fig.show()


def return_over_episodes(data_dataframe, model_name, plot_path,
                         save_plot=True, filter_return=True, size=50000, save_html=False):
    """

    Args:
        save_html:
        filter_return:
        size:
        save_plot:
        data_dataframe:
        model_name:
        plot_path:
    """

    if filter_return:
        data_dataframe = data_dataframe.rolling(window=size, min_periods=1, center=True).mean()

    title = f'Return of {model_name} for {len(data_dataframe)} episodes, moving average over {size} episodes'

    x_axis_label = 'Episode'
    y_axis_label = 'Return'
    fig = px.line(data_dataframe,
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


def steps_over_episodes(data_dataframe, model_name, plot_path,
                        save_plot=True, save_html=False):
    """

    Args:
        save_html:
        save_plot:
        data_dataframe:
        model_name:
        plot_path:
    """

    title = f'Steps of {model_name} for every episode'
    x_axis_label = 'Episode'
    y_axis_label = 'Steps per episode'
    fig = px.line(data_dataframe,
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


def multiple_plots(data_dataframe, plot_path, save_plot=True, save_html=False):

    title = 'test'
    fig = px.histogram(data_dataframe, title=title)
    # fig.write_image(plot_path)
    fig.update_layout(barmode='group')
    fig.show()
    print('d')


def create_all_plots(model_name, data_dataframe, plot_base_path, save_plots, filter_return, filter_size, save_html, split):
    """

    Args:
        model_name:
        save_html:
        data_dataframe:
        plot_base_path:
        save_plots:
        filter_return: bool, Sets whether to apply a moving average to the return of the model
        filter_size:
        split:
    """
    print('.... creating return histogram')
    title = f'Histogram of the return per episode for {model_name}'
    plot_path = os.path.join(plot_base_path, 'return_histogram.png')
    create_histogram(data_dataframe['model_return'], title, plot_path, save_plot=save_plots, save_html=save_html)

    print('.... creating room guesses histogram')
    title = f'Histogram of room guesses for {model_name}'
    plot_path = os.path.join(plot_base_path, 'hits_histogram.png')
    create_histogram(data_dataframe['model_hits'], title, plot_path, save_plot=save_plots, save_html=save_html)

    print('.... creating steps histogram')
    title = f'Histogram of number of steps per episode for {model_name}'
    plot_path = os.path.join(plot_base_path, 'steps_histogram.png')
    create_histogram(data_dataframe['model_steps'], title, plot_path, save_plot=save_plots, save_html=save_html)

    print('.... plotting return per episode')
    plot_path = os.path.join(plot_base_path, 'return_over_episodes.png')
    return_over_episodes(data_dataframe['model_return'], model_name, plot_path, save_plot=save_plots,
                         filter_return=filter_return, size=filter_size, save_html=save_html)

    print('.... plotting steps per episode')
    plot_path = os.path.join(plot_base_path, 'steps_over_episodes.png')
    steps_over_episodes(data_dataframe['model_steps'], model_name, plot_path, save_plot=save_plots, save_html=save_html)

    print('.... plotting accuracy per episode')
    plot_path = os.path.join(plot_base_path, 'accuracy.png')
    plot_accuracy(data_dataframe['model_hits'], model_name, plot_path, split, save_plot=save_plots, save_html=save_html)


