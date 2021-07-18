import numpy as np
import os
import plotly.express as px
import pandas as pd


def create_histogram(data, title):
    # TODO better title
    title = 'Counts of {}'.format(title)

    df = pd.DataFrame(data)
    fig = px.histogram(df, title=title)
    # TODO save figure with distinct name
    fig.show()


def create_figure(model_steps, model_return, model_name, base_path):
    """

    Args:
        model_steps:
        model_return:
        model_name:
        base_path:
    """
    title = 'Return of {} over {} episodes'.format(model_name, len(model_return))
    x_axis_label = 'Steps'
    y_axis_label = 'Return'
    fig = px.line(x=np.cumsum(model_steps),
                  y=model_return,
                  title=title,
                  )
    fig.update_xaxes(title_text=x_axis_label)
    fig.update_yaxes(title_text=y_axis_label)
    # TODO better name for plot
    fig.write_image(os.path.join(base_path, 'fig.png'))
    # TODO save HTML plot
