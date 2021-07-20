import numpy as np
import plotly.express as px
import pandas as pd
from scipy.ndimage.filters import uniform_filter1d


def create_histogram(data, title, plot_path, save_plot=True):
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
