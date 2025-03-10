import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
pd.options.plotting.backend = "plotly"


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

    print('shape model_return, model_hits, model_steps: ', np.shape(model_return), np.shape(model_hits), np.shape(model_steps))

    with open(os.path.join(base_path, 'model_parameters.json'), 'r') as fp:
        parameters = json.load(fp)
    num_episodes = parameters['training']['num_episodes']

    model_name = [n for n in parameters.keys() if n != 'MapWorld' and n != 'training'][0] + os.path.split(base_path)[1][10:]

    plot_base_path = os.path.join(base_path, 'plots')

    data_dict = {'model_return': model_return, 'model_steps': model_steps, 'model_hits': model_hits}
    data_dataframe = pd.DataFrame(data_dict)
    return data_dataframe, num_episodes, plot_base_path, model_name


def aggregate_and_plot_r1_data(split, filter_return, size):
    """
    Loads the data. Assumes base_path is structured like: "results/**/**/raw"
    Args:
        base_path: string, path to folder where the data is saved to
    Returns: Results (return, steps, hits) as numpy arrays,
    """
    mr_list = []
    # mh_list = []
    ms_list = []
    acc_list = []
    for p in ['results/actor_critic/2022-03-28_r1_masked_2M',
              'results/actor_critic/2022-04-27_r1_masked_1',
              'results/actor_critic/2022-04-29_r1_masked_2',
              'results/actor_critic/2022-04-30_r1_masked_3',
              'results/actor_critic/2021-11-01_r1_masked']:
        model_return = np.load(os.path.join(p, 'model_return.npy'))
        model_hits = np.load(os.path.join(p, 'model_hits.npy'))
        model_steps = np.load(os.path.join(p, 'model_steps.npy'))
        mr_list.append(model_return)
        # mh_list.append(model_hits)
        ms_list.append(model_steps)
        accuracy_per_split, step, _ = compute_split_accuracy(model_hits, split)
        acc_list.append(accuracy_per_split)

    # print('shape mr, mh, ms: ', np.shape(mr_list), np.shape(mh_list), np.shape(ms_list))
    model_return = np.mean(mr_list, axis=0)
    # model_hits = np.mean(mh_list, axis=0)
    model_steps = np.mean(ms_list, axis=0)
    acc_list = np.mean(acc_list, axis=0)

    print('shape model_return, model_hits, model_steps: ', np.shape(model_return), np.shape(model_hits), np.shape(model_steps))

    with open(os.path.join(p, 'model_parameters.json'), 'r') as fp:
        parameters = json.load(fp)
    # num_episodes = parameters['training']['num_episodes']

    # model_name = [n for n in parameters.keys() if n != 'MapWorld' and n != 'training'][0] + os.path.split(p)[1][10:]

    data_dict = {'model_return': model_return, 'model_steps': model_steps, 'model_hits': model_hits}
    data_dataframe = pd.DataFrame(data_dict)

    fig = px.line(x=range(0, len(model_hits), step),
                  y=acc_list)

    legend = f'Mean accuracy: {np.round(np.mean(accuracy_per_split), decimals=4)}'
    fig.update_xaxes(title_text='Episode', showgrid=False, linecolor="#BCCCDC")
    fig.update_yaxes(title_text='Accuracy', showgrid=False, linecolor="#BCCCDC")
    fig.update_layout(plot_bgcolor='#FFF', title=legend)
    fig.write_image('results/aggregated_accuracy_r1.png', scale=2.0)

    if filter_return:
        data_dataframe = data_dataframe.rolling(window=size, min_periods=1, center=True).mean()

    x_axis_label = 'Episode'
    y_axis_label = 'Reward'
    fig = px.line(data_dataframe['model_return'])
    fig.update_xaxes(title_text=x_axis_label, showgrid=False, linecolor="#BCCCDC")
    fig.update_yaxes(title_text=y_axis_label, showgrid=False, linecolor="#BCCCDC")
    fig.update_layout(plot_bgcolor='#FFF')

    fig.write_image('results/aggregated_reward_r1.png', scale=2.0)


def compute_split_accuracy(model_hits, split=100):
    """

    Args:
        model_hits:
        split:

    Returns:

    """
    # compute total accuracy to four decimals
    accuracy = np.round(np.sum(model_hits)/(len(model_hits)), decimals=4)

    # create splits of even length of the data
    split_model_hits = np.array_split(model_hits, split)
    split_size = len(split_model_hits[0])
    # compute accuracy over every split
    accuracy_per_split = [np.sum(x)/len(x) for x in split_model_hits]
    steps = int(len(model_hits) / len(split_model_hits))

    return accuracy_per_split, steps, accuracy


def plot_individual_accuracy(model_hits, plot_path, split=100, save_plot=True, save_html=False):
    """

    Args:
        model_hits:
        plot_path:
        split:
        save_plot:
        save_html:
    """

    accuracy_per_split, step, _ = compute_split_accuracy(model_hits, split)

    x_axis_label = 'Episode'
    y_axis_label = 'Accuracy'

    fig = px.line(x=range(0, len(model_hits), step),
                  y=accuracy_per_split)

    fig.update_xaxes(title_text=x_axis_label, showgrid=False, linecolor="#BCCCDC")
    fig.update_yaxes(title_text=y_axis_label, showgrid=False, linecolor="#BCCCDC")
    fig.update_layout(plot_bgcolor='#FFF')
    if save_plot:
        fig.write_image(plot_path, scale=2.0)
        if save_html:
            html_path = plot_path[:-4] + '.html'
            fig.write_html(html_path)
    else:
        fig.show()


def create_histogram(data_dataframe, x_axis_label, plot_path='', save_plot=True, save_html=False):
    """

    Args:
        x_axis_label:
        save_html:
        data_dataframe:
        plot_path:
        save_plot:
    """

    # TODO where are the axis descriptions ?
    fig = px.histogram(data_dataframe)
    fig.update_xaxes(title_text=x_axis_label, showgrid=False, linecolor="#BCCCDC")
    fig.update_yaxes(showgrid=False, linecolor="#BCCCDC")
    fig.update_layout(plot_bgcolor='#FFF')
    if save_plot:
        fig.write_image(plot_path, scale=2.0)
        if save_html:
            html_path = plot_path[:-4] + '.html'
            fig.write_html(html_path)
    else:
        fig.show()


def return_over_episodes(data_dataframe, plot_path,
                         save_plot=True, filter_return=True, size=50000, save_html=False):
    """

    Args:
        save_html:
        filter_return:
        size:
        save_plot:
        data_dataframe:
        plot_path:
    """

    if filter_return:
        data_dataframe = data_dataframe.rolling(window=size, min_periods=1, center=True).mean()

    x_axis_label = 'Episode'
    y_axis_label = 'Reward'
    fig = px.line(data_dataframe)
    fig.update_xaxes(title_text=x_axis_label, showgrid=False, linecolor="#BCCCDC")
    fig.update_yaxes(title_text=y_axis_label, showgrid=False, linecolor="#BCCCDC")
    fig.update_layout(plot_bgcolor='#FFF')

    if save_plot:
        fig.write_image(plot_path, scale=2.0)
        if save_html:
            html_path = plot_path[:-4] + '.html'
            fig.write_html(html_path)
    else:
        fig.show()


def steps_over_episodes(data_dataframe, plot_path,
                        save_plot=True, save_html=False):
    """

    Args:
        save_html:
        save_plot:
        data_dataframe:
        plot_path:
    """

    # title = f'Steps of {model_name} for every episode'
    x_axis_label = 'Episode'
    y_axis_label = 'Steps per episode'
    fig = px.line(data_dataframe)
    fig.update_xaxes(title_text=x_axis_label, showgrid=False, linecolor="#BCCCDC")
    fig.update_yaxes(title_text=y_axis_label, showgrid=False, linecolor="#BCCCDC")
    fig.update_layout(plot_bgcolor='#FFF')

    if save_plot:
        fig.write_image(plot_path, scale=2.0)
        if save_html:
            html_path = plot_path[:-4] + '.html'
            fig.write_html(html_path)
    else:
        fig.show()


def plot_group_accuracy(data_dataframe, names, step, plot_path, accuracies, save_plot=True, save_html=True):
    """
    Creates a plot of the change in accuracy over all experiments.
    Args:
        accuracies:
        save_plot: string,
        save_html: string,
        names: list, reward function names
        data_dataframe: list, containing accuracy per every "step" episode
        plot_path: string,
        step: int,
    """
    # add a new column to the df for the episode corresponding to each accuracy, important for the plot
    data_dataframe.append(list(range(0, np.shape(data_dataframe)[1]*step, step)))

    names.append('Episode')
    df = pd.DataFrame(data_dataframe, names).transpose()
    df.set_index('Episode', inplace=True, drop=True)

    legend = f'Mean accuracy: {np.round(np.mean(accuracies), decimals=4)}'

    fig = df.plot()
    x_axis_label = 'Episode'
    y_axis_label = 'Accuracy'
    fig.update_xaxes(title_text=x_axis_label, showgrid=False, linecolor="#BCCCDC")
    fig.update_yaxes(title_text=y_axis_label, showgrid=False, linecolor="#BCCCDC")
    fig.update_layout(plot_bgcolor='#FFF', title=legend)
    if save_plot:
        fig.write_image(plot_path, scale=2.0)
        if save_html:
            html_path = plot_path[:-4] + '.html'
            fig.write_html(html_path)
    else:
        fig.show()


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
    plot_path = os.path.join(plot_base_path, f'{model_name}_reward_histogram.png')
    create_histogram(data_dataframe['model_return'], 'Reward', plot_path, save_plot=save_plots, save_html=save_html)

    print('.... creating room guesses histogram')
    plot_path = os.path.join(plot_base_path, f'{model_name}_hits_histogram.png')
    create_histogram(data_dataframe['model_hits'], 'Hits', plot_path, save_plot=save_plots, save_html=save_html)

    print('.... creating steps histogram')
    plot_path = os.path.join(plot_base_path, f'{model_name}_steps_histogram.png')
    create_histogram(data_dataframe['model_steps'], 'Steps', plot_path, save_plot=save_plots, save_html=save_html)

    print('.... plotting return per episode')
    plot_path = os.path.join(plot_base_path, f'{model_name}_reward_over_episodes.png')
    return_over_episodes(data_dataframe['model_return'], plot_path, save_plot=save_plots,
                         filter_return=filter_return, size=filter_size, save_html=save_html)

    print('.... plotting steps per episode')
    plot_path = os.path.join(plot_base_path, f'{model_name}_steps_over_episodes.png')
    steps_over_episodes(data_dataframe['model_steps'], plot_path, save_plot=save_plots, save_html=save_html)

    print('.... plotting accuracy per episode')
    plot_path = os.path.join(plot_base_path, f'{model_name}_accuracy.png')
    plot_individual_accuracy(data_dataframe['model_hits'], plot_path, split, save_plot=save_plots, save_html=save_html)


