#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from typing import Union, Tuple

import numpy as np
from numpy import ndarray


def cross_entropy_loss(true_labels: Union[List[int], np.ndarray], predicted_probs: Union[List[float], np.ndarray]) -> Tuple[float, np.ndarray]:
    """Calculate the cross-entropy loss between true labels and predicted probabilities.  Categorical 
    cross-entropy is explicitly used to compare a “ground-truth” probability (y or "targets") and
    some predicted distribution (y-hat or "predictions"), so it makes sense to use cross-entropy here.
    It is also one of the most commonly used loss functions with a ``softmax activation`` on the output layer.
    
    The formula for calculating the categorical cross-entropy of ``y`` (actual/desired distribution) and
    ``y-hat`` (predicted distribution) is:

    .. math::
        L(y, \\hat{y}) = -\\sum_{i} y_i_j \\log(\\hat{y}_i_j)

    where:
        - :math:`L` is the cross-entropy loss.
        - :math:`y_i_j` is the actual probability distribution.
        - :math:`\\hat{y}_i_j` is the predicted probability distribution.

    :param true_labels: (Union[List[int], np.ndarray]): A list or numpy array of true labels (0 or 1).
    :param predicted_probs: (Union[List[float], np.ndarray]): A list or numpy array of predicted
    probabilities (between 0 and 1).
    :return: (Tuple[float, np.ndarray]): The average cross-entropy loss and the array of individual losses.
    """
    # Convert lists to numpy arrays if necessary
    if isinstance(true_labels, list):
        true_labels = np.array(true_labels)
    if isinstance(predicted_probs, list):
        predicted_probs = np.array(predicted_probs)

    # Attempt to broadcast arrays to compatible shapes
    try:
        # check is there is a shape mismatch
        if true_labels.shape != predicted_probs.shape:
            true_labels, predicted_probs = np.broadcast_arrays(true_labels, predicted_probs)
    except ValueError:
        raise ValueError(f"Shape mismatch: true_labels shape {true_labels.shape} and predicted_probs shape {predicted_probs.shape} cannot be broadcasted to the same shape.")

    # Replace nan values in predicted_probs with 0 or 1
    predicted_probs = np.nan_to_num(predicted_probs, nan=0, posinf=1.0, neginf=0.0, copy=True)

    # Calculate the cross-entropy loss for each sample
    losses = - (
            true_labels * np.log(predicted_probs)
            + (1 - true_labels)
            * np.log(1 - predicted_probs)
        )

    # Calculate the average cross-entropy loss
    avg_loss = np.mean(losses)
    return avg_loss, losses


def sort_cross_entropy_loss(__predicted_probs: np.ndarray, __cross_entropy_losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort the cross-entropy losses based on the probability predictions.

    :param __predicted_probs: (np.ndarray): A numpy array of predicted probabilities.
    :param __cross_entropy_losses: (np.ndarray): A numpy array of cross-entropy losses.
    :returns tuple[np.ndarray, np.ndarray]: Two numpy arrays, one for sorted predicted
    probabilities and one for sorted cross-entropy losses.
    """
    __sorted_dict = dict(sorted(zip(__predicted_probs, __cross_entropy_losses)))
    __sorted_probs = np.array(list(__sorted_dict.keys()))
    __sorted_losses = np.array(list(__sorted_dict.values()))
    return __sorted_probs, __sorted_losses


def mean_square_error(y_true: ndarray, y_pred: ndarray) -> float:
    """
    Calculate the Mean Square Error (MSE) loss between the true and predicted values.

    :param y_true: (ndarray) - The true labels or values.
    :param y_pred: (ndarray) - The predicted labels or values.
    :returns: float: The mean square error loss.
    """
    # Calculate the mean of the squared differences
    __mse = np.mean((y_pred - y_true) ** 2)
    print(__mse)
    return __mse