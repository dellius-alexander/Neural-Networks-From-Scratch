#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.ndimage import zoom

from src.functions.activation import Activation, Softmax, ReLU
from src.layer.dense import Dense
from src.utils.logger import getLogger

log = getLogger(__name__)


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    activation: Activation,
    epochs: int,
    learning_rate: float,
):
    # Set seed for reproducibility
    np.random.seed(42)

    # Initialize weights and bias with random values
    weights = np.random.rand(features.shape[1])
    bias = np.random.rand(1)

    # Initialize list to store error values for each epoch of training
    error_sum = []

    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        # Forward pass: compute the weighted sum of inputs and apply the activation function
        weighted_sum_of_inputs = np.dot(features, weights) + bias
        # Apply the activation function to the weighted sum of inputs
        predictions = activation(weighted_sum_of_inputs)
        # Compute the error as the difference between the predictions and actual values of the labels in our dataset.
        error = predictions - labels
        # Compute the sum of errors for this epoch for each individual data record,
        # and append it to the error_sum list.
        error_sum.append(np.sum(error))

        # Typically, for classification models, the cost is the cross-entropy or log loss. Instead, we'll use the mean square error because that makes our gradient computation simple. So, the derivative of the cost function with respect to the prediction and that is simply the error that we've computed earlier.
        dcost_dpred = error

        # The derivative of the predictions is simply the derivative of the output of the activation function with respect to the input to the activation function. We compute the derivative of the activation.function with respect to z, giving us d_pred/dz.
        dpred_dz = activation.derivative(predictions)

        # Now, the gradients are the d_cost/d_pred multiplied by d_pred/dz. Now, all that's left to compute is the derivative of predictions with respect to weights, the model parameters.
        gradients = dcost_dpred * dpred_dz

        # The derivative of z with respect to the model parameters, where z is the output of the linear transformation, is simply the features themselves. So that's why we have, features.T, which I then multiply by gradients. This dot product completes the multiplication that we have here in this chain rule, giving us d_cost/dw. Now that, we have the partial derivative which gives us the gradient vector, I multiply that by the learning_rate, this is again on line 25, and I subtract this from the current value of weights, to give us the weights for the next iteration of the model
        weights -= learning_rate * np.dot(features.T, gradients)

        # For each gradient in the gradient vector, multiply that by the learning rate, and use that to find the updated value of the bias term. Since the bias term does not correspond to a feature coefficient. We do not need to multiply it by the feature vector. This is essentially the training process.
        for gradient in gradients:
            bias -= learning_rate * gradient

    return weights, bias, error_sum


"""Parallel Training of Neural Networks with Ray Tune."""


def train_epoch(
    epoch: int,
    X: np.ndarray,
    y: np.ndarray,
    dense: Dense,
    dense2: Dense,
    relu: ReLU,
    softmax: Softmax,
) -> tuple:
    """Train the model for one epoch and return the epoch number,
    loss, accuracy, weights, and biases.

    :param epoch: (int): The epoch number
    :param X: (np.ndarray): The input data
    :param y: (np.ndarray): The ground truth labels
    :param dense: (Dense): The first dense layer
    :param dense2: (Dense): The second dense layer
    :param relu: (ReLU): The ReLU activation function
    :param softmax: (Softmax): The softmax activation function
    :return: (tuple): The epoch number, loss, accuracy, weights, and biases
    """
    # Generate a new set of weights and biases for each iteration
    dense.weights = 0.5 * np.random.randn(
        dense.weights.shape[0], dense.weights.shape[1]
    )
    dense.biases = 0.1 * np.random.randn(dense.biases.shape[0], dense.biases.shape[1])
    dense2.weights = 0.5 * np.random.randn(
        dense2.weights.shape[0], dense2.weights.shape[1]
    )
    dense2.biases = 0.1 * np.random.randn(
        dense2.biases.shape[0], dense2.biases.shape[1]
    )
    y_copy = y.copy()

    # Forward pass
    dense.forward(X)
    dense.activation = relu(dense.output)
    dense2.forward(dense.activation)
    predictions = softmax(dense2.output)

    # Reshape the ground truth labels if transformed during the forward pass
    if len(y_copy.shape) == 1:
        y_copy = np.array([y_copy])
    elif len(y_copy.shape) > 2:
        y_copy = np.array([y_copy]).reshape(-1, 1)

    # Match the size of predictions to the size of y
    if (y_copy.shape[1], y_copy.shape[0]) != predictions.shape or (
        y_copy.shape[0],
        y_copy.shape[1],
    ) != predictions.shape:
        zoom_factor = np.array(y_copy.shape) / np.array(predictions.shape)
        predictions = zoom(predictions, zoom_factor, order=3)

    # Calculate the loss
    avg_loss, loss = dense2.loss(predictions, y_copy)

    # One-hot encode the predictions
    if len(predictions.shape) >= 2:
        predictions_class = np.argmax(predictions, axis=1)
    elif len(predictions.shape) == 1:
        predictions_class = np.argmax(predictions, axis=0)
    else:
        predictions_class = np.argmax(predictions)

    # One-hot encode the ground truth labels
    if len(y_copy.shape) >= 2:
        y_copy = np.argmax(y_copy, axis=1)
    elif len(y_copy.shape) == 1:
        y_copy = np.argmax(y_copy, axis=0)
    else:
        y_copy = np.argmax(y_copy)

    # Calculate the accuracy
    accuracy = np.mean(predictions_class == y_copy)
    log.info(f"Epoch: {epoch}, Loss: {avg_loss:.7f}, Accuracy: {accuracy:.7f}")
    return (
        epoch,
        avg_loss,
        accuracy,
        dense.weights.copy(),
        dense.biases.copy(),
        dense2.weights.copy(),
        dense2.biases.copy(),
    )


def parallel_training(
    epochs: int,
    X: np.ndarray,
    y: np.ndarray,
    dense: Dense,
    dense2: Dense,
    relu: ReLU,
    softmax: Softmax,
) -> tuple:
    """Train the model in parallel using ProcessPoolExecutor and return the
    best epoch, loss, accuracy, weights, and biases.

    :param epochs: (int): The number of epochs
    :param X: (np.ndarray): The input data
    :param y: (np.ndarray): The ground truth labels
    :param dense: (Dense): The first dense layer
    :param dense2: (Dense): The second dense layer
    :param relu: (ReLU): The ReLU activation function
    :param softmax: (Softmax): The softmax activation function
    :return: (tuple): The best epoch, loss, accuracy, weights, and biases
    """
    # Create variables to store the best model metrics
    lowest_loss = np.inf
    best_accuracy = 0
    best_epoch = 0
    best_weights = np.array([])
    best_biases = np.array([])
    best_weights2 = np.array([])
    best_biases2 = np.array([])
    cpu_count = os.cpu_count()

    # Train the model in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        # Submit the training tasks to the executor
        futures = [
            executor.submit(train_epoch, epoch, X, y, dense, dense2, relu, softmax)
            for epoch in range(epochs)
        ]
        # Iterate over the completed futures to find the best model metrics
        for future in as_completed(futures):
            (
                __best_epoch,
                __lowest_loss,
                __best_accuracy,
                __best_weights,
                __best_biases,
                __best_weights2,
                __best_biases2,
            ) = future.result()
            # Evaluate the best model based on the lowest loss and highest accuracy
            if __lowest_loss < lowest_loss or __best_accuracy > best_accuracy:
                lowest_loss = __lowest_loss
                best_accuracy = __best_accuracy
                best_epoch = __best_epoch
                best_weights = __best_weights
                best_biases = __best_biases
                best_weights2 = __best_weights2
                best_biases2 = __best_biases2
    return (
        best_epoch,
        lowest_loss,
        best_accuracy,
        best_weights,
        best_biases,
        best_weights2,
        best_biases2,
    )
