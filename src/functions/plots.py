import numpy as np
import plotly.graph_objects as go

from typing import Annotated, Any, Tuple

def plot_error(__error_arr: Annotated[np.ndarray, 'error_sum']) -> Tuple[go.Figure, Any]:
    """Plot the error values against the epochs.
    :param __error_arr: np.ndarray: The array of error values for each epoch
    :return: go.Figure: The plotly figure object
    """
    __fig = go.Figure()
    __x = np.arange(1, len(__error_arr) + 1)
    __fig.add_trace(
        go.Scatter(
            x=__x, y=__error_arr, mode='lines+markers', name='Error', line=dict(color='blue'),
            marker=dict(color='blue', size=8), showlegend=True
        ))
    __fig.update_layout(
        title='Error vs. Epochs',
        xaxis_title='Epochs',
        yaxis_title='Error',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        xaxis=dict({"showgrid": True, "color": 'white'}),
        yaxis=dict({"showgrid": True, "color": 'white'}),
        title_font=dict({"size": 20, "color": 'white'}),
        legend=dict(bgcolor='black', bordercolor='white', font=dict(color='white'))
    )
    __results = {
        'epochs': __x,
        'error': __error_arr
    }
    return __fig, __results

