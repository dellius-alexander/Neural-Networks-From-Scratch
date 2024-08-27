#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Any
import numpy as np
from numpy import ndarray, dtype


def cross_entropy_loss(true_labels: List[int], predicted_probs: List[float]) -> tuple[Any, ndarray[Any, dtype[Any]]]:
    """Calculate the cross-entropy loss between true labels and predicted probabilities.

    :param true_labels: (List[int]): A list of true labels (0 or 1).
    :param predicted_probs: (List[float]): A list of predicted probabilities (between 0 and 1).
    :return: (tuple[Any, ndarray[Any, dtype[Any]]]): The cross-entropy loss between the true labels and predicted probabilities.
    """
    # Calculate the cross-entropy loss for each sample
    __losses = []
    for p, q in zip(true_labels, predicted_probs):
        __loss = - (p * np.log(q) + (1 - p) * np.log(1 - q))
        __losses.append(__loss)
        # print(f"True Label: {p}, Predicted Probability: {q}, Cross-Entropy Loss: {__loss:.4f}")
    # Calculate the average cross-entropy loss
    __avg_loss = np.mean(__losses)
    return __avg_loss, np.array(__losses)


def sort_cross_entropy_loss(__predicted_probs: np.ndarray, __cross_entropy_losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort the cross-entropy losses based on the probability predictions.


    :param __predicted_probs: (np.ndarray): A numpy array of predicted probabilities.
    :param __cross_entropy_losses: (np.ndarray): A numpy array of cross-entropy losses.

    :returns tuple[np.ndarray, np.ndarray]: Two numpy arrays, one for sorted predicted probabilities and one for sorted cross-entropy losses.
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