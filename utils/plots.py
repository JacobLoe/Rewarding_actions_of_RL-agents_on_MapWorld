import numpy as np
import plotly.express as px
import pandas as pd
from scipy.ndimage.filters import uniform_filter1d
import os


def create_histogram(data, title, plot_path='', save_plot=False):
    """

    Args:
        data:
        title:
        plot_path:
        save_plot:
    """
    # TODO better title
    title = 'Counts of {}'.format(title)

    df = pd.DataFrame(data)
    fig = px.histogram(df, title=title)
    if save_plot:
        fig.write_image(plot_path)
        html_path = plot_path[:-4] + '.html'
        fig.write_html(html_path)
    else:
        fig.show()


def create_figure(model_steps, model_return, model_name, plot_path, save_plot=True, filter_return=True, size=100):
    """

    Args:
        filter_return:
        size:
        save_plot:
        model_steps:
        model_return:
        model_name:
        plot_path:
    """

    if filter_return:
        mreturn = uniform_filter1d(model_return, mode='constant', size=size)
    else:
        mreturn = model_return

    title = 'Return of {} for {} episodes, moving average over {} episodes'.format(model_name, len(model_return), size)
    x_axis_label = 'Steps'
    y_axis_label = 'Return'
    fig = px.line(x=np.cumsum(model_steps),
                  y=mreturn,
                  title=title,
                  )
    fig.update_xaxes(title_text=x_axis_label)
    fig.update_yaxes(title_text=y_axis_label)
    if save_plot:
        fig.write_image(plot_path)
        # TODO save HTML plot
    else:
        fig.show()


def create_all_plots(model_return, model_steps, model_hits, num_episodes,
                     plot_base_path, save_plots, filter_return, filter_size):
    """

    Args:
        model_return:
        model_steps:
        model_hits:
        num_episodes:
        plot_base_path:
        save_plots:
        filter_return: bool, Sets whether to apply a moving average to the return of the model
        filter_size:
    """
    title = 'the return over {}'.format(num_episodes)
    plot_path = os.path.join(plot_base_path, 'return_histogram.png')
    create_histogram(model_return, title, plot_path, save_plot=save_plots)

    title = 'room guesses over {}'.format(num_episodes)
    plot_path = os.path.join(plot_base_path, 'hits_histogram.png')
    create_histogram(model_hits, title, plot_path, save_plot=save_plots)

    title = 'the steps over {}'.format(num_episodes)
    plot_path = os.path.join(plot_base_path, 'steps_histogram.png')
    create_histogram(model_steps, title, plot_path, save_plot=save_plots)

    plot_path = os.path.join(plot_base_path, 'return_over_episodes.png')
    create_figure(model_steps, model_return, 'REINFORCE', plot_path, save_plot=save_plots,
                  filter_return=filter_return, size=filter_size)
