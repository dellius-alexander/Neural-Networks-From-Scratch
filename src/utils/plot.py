#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import plotly.graph_objects as go

from typing import Annotated, Any, Tuple


def plot_error(
    __error_arr: Annotated[np.ndarray, "error_sum"]
) -> Tuple[go.Figure, Any]:
    """Plot the error values against the epochs.
    :param __error_arr: np.ndarray: The array of error values for each epoch
    :return: go.Figure: The plotly figure object
    """
    __fig = go.Figure()
    __x = np.arange(1, len(__error_arr) + 1)
    __fig.add_trace(
        go.Scatter(
            x=__x,
            y=__error_arr,
            mode="lines+markers",
            name="Error",
            line=dict(color="blue"),
            marker=dict(color="blue", size=8),
            showlegend=True,
        )
    )
    __fig.update_layout(
        title="Error vs. Epochs",
        xaxis_title="Epochs",
        yaxis_title="Error",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        xaxis=dict({"showgrid": True, "color": "white"}),
        yaxis=dict({"showgrid": True, "color": "white"}),
        title_font=dict({"size": 20, "color": "white"}),
        legend=dict(bgcolor="black", bordercolor="white", font=dict(color="white")),
    )
    __results = {"epochs": __x, "error": __error_arr}
    return __fig, __results


def get_camera_view(perspective: str):
    """Set the camera view (orientation).

    Options include: 'low-angle-side', 'high-angle-front', 'rotated-top-down', 'top-down', 'side-view', 'diagonal',
        'zoom-out', 'zoom-in', 'custom'

    Usage:

    >>> camera = get_camera_view(perspective='low-angle-side')


    :param perspective: str: The perspective to set the camera view to.
    :return: dict: The camera view orientation
    """
    # TODO: Set the camera view (orientation)
    # 1. Low Angle Side View (Looking Up at the Plot):
    # This view gives the impression of looking up at the plot from a low angle.
    if perspective == "low-angle-side":
        return dict(eye=dict(x=1.5, y=0.5, z=0.5))

    # 2. High Angle Front View (Looking Down on the Plot):
    # This view looks down at the plot from a high angle, emphasizing the z-axis.
    elif perspective == "high-angle-front":
        return dict(eye=dict(x=0.5, y=0.5, z=2.5))

    # 3. Rotated Top-Down View:
    # This view looks down from above but rotated along the y-axis,
    # giving a skewed top-down perspective.
    elif perspective == "rotated-top-down":
        return dict(eye=dict(x=2.0, y=2.0, z=1.0))

    # 4. Top-down view (looking down the z-axis):
    # This view looks straight down along the z-axis.
    elif perspective == "top-down":
        return dict(eye=dict(x=0.0, y=0.0, z=2.5))

    # 5. Side view (looking along the x-axis):
    # This view looks from the side along the x-axis.
    elif perspective == "side-view":
        return dict(eye=dict(x=2.5, y=0.0, z=0.0))

    # 6. Diagonal view (default perspective):
    # This view looks from a diagonal angle, giving a balanced perspective.
    elif perspective == "diagonal":
        return dict(eye=dict(x=1.5, y=1.5, z=1.5))
    # camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))

    # 7. # Higher values to zoom out:
    # Zoom out by increasing the distance of the camera (increase x, y, z values)
    elif perspective == "zoom-out":
        return dict(eye=dict(x=3.0, y=3.0, z=3.0))

    # 8. # Lower values to zoom in:
    # Zoom in by decreasing the distance of the camera (decrease x, y, z values)
    elif perspective == "zoom-in":
        return dict(eye=dict(x=0.5, y=0.5, z=0.5))

    # 9. # Custom view:
    # Set the camera view to a custom position
    elif perspective == "custom":
        return dict(eye=dict(x=1.5, y=1.5, z=1.5))

    # Default view
    else:
        return dict(eye=dict(x=1.5, y=1.5, z=1.5))
