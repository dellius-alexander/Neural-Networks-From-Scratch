#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABC

# TODO: Annotated added in python version 3.9 and above
from typing import Annotated, Any, List

import numpy as np
import pandas as pd

from src.functions.loss import cross_entropy_loss
from src.utils.logger import getLogger

log = getLogger(__name__)

N_Inputs = Annotated[np.ndarray, "The number of inputs to the layer"]
N_Neurons = Annotated[np.ndarray, "The number of neurons in the layer"]
Matrix = Annotated[List, np.ndarray, "A matrix of values"]


class IDense(ABC):
    inputs: Annotated[np.ndarray, "The inputs to the layer"]
    weights: Annotated[np.ndarray, "The weights of the layer"]
    biases: Annotated[np.ndarray, "The biases of the layer"]
    output: Annotated[np.ndarray, "The output values of the layer"] = None
    dinputs: Annotated[
        np.ndarray, "The gradient of the loss with respect to the inputs"
    ] = None
    dweights: Annotated[np.ndarray, "The gradients of the weights of the layer"] = None
    dbiases: Annotated[np.ndarray, "The gradients of the biases of the layer"] = None
    activation: Annotated[Any, "The activation function of the layer"] = None

    @abstractmethod
    def __init__(self, n_inputs: int, n_neurons: int):
        """
        Initialize the layers weights and biases using the number of inputs and neurons

        :param n_inputs: int: The number of inputs to the layer
        :param n_neurons: int: The number of neurons in the layer
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: N_Inputs) -> N_Neurons:
        """
        Forward pass of the layer.
        Calculate output values from inputs, weights and biases
        :param inputs: N_Inputs: The inputs to the layer
        :return: N_Neurons: The output values
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, dvalues: N_Neurons, learning_rate: float) -> N_Inputs:
        """
        Backward pass of the layer.
        Calculate the gradient of the loss with respect to the inputs
        :param dvalues: N_Neurons: The gradient of the loss with respect to the output values
        :return: N_Inputs: The gradient of the loss with respect to the inputs
        """
        raise NotImplementedError

    @abstractmethod
    def params(self) -> tuple:
        """
        Get the weights and biases of the layer
        :return: tuple: The weights and biases
        """
        raise NotImplementedError

    @abstractmethod
    def grads(self) -> tuple:
        """
        Get the gradients of the weights and biases of the layer
        :return: tuple: The gradients of the weights and biases
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, y_pred: N_Neurons, y_true: N_Neurons) -> tuple:
        """
        Calculate the loss of the layer
        :param y_pred: N_Neurons: The predicted values
        :param y_true: N_Neurons: The true values
        :return: float: The loss value
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """
        Get the string representation of the layer
        :return: str: The string representation of the layer
        """
        raise NotImplementedError


class Dense(IDense):
    def __init__(self, n_inputs: int, n_neurons: int):
        """
        Initialize the layers weights and biases using the number of inputs and neurons

        :param n_inputs: int: The number of inputs to the layer
        :param n_neurons: int: The number of neurons in the layer
        """
        self.weights = 0.05 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: N_Inputs) -> N_Neurons:
        """
        Forward pass of the layer.
        Calculate output values from inputs, weights and biases
        :param inputs: N_Inputs: The inputs to the layer
        :return: N_Neurons: The output values
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues: N_Neurons, learning_rate: float) -> N_Inputs:
        """
        Perform the backward pass and update weights and biases.

        :param dvalues: N_Neurons: The gradient of the loss with respect to the outputs
        :param learning_rate: float: The learning rate
        :return: N_Inputs: The gradient of the loss with respect to the inputs
        """
        # Gradient of the loss with respect to weights
        self.dweights = np.dot(self.inputs.T, dvalues)

        # Gradient of the loss with respect to biases
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient of the loss with respect to inputs
        self.dinputs = np.dot(dvalues, self.weights.T)

        # Update weights and biases using the learning rate
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

        return self.dinputs

    def params(self) -> tuple:
        """
        Get the weights and biases of the layer
        :return: tuple: The weights and biases
        """
        return self.weights, self.biases

    def grads(self) -> tuple:
        """
        Get the gradients of the weights and biases of the layer
        :return: tuple: The gradients of the weights and biases
        """
        return self.dweights, self.dbiases

    def loss(self, y_pred: N_Neurons, y_true: N_Neurons) -> tuple:
        """
        Calculate the loss of the layer
        :param y_pred: (N_Neurons): The predicted values
        :param y_true: (N_Neurons): The true values
        :return: (float): The loss value
        """
        if isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
            return cross_entropy_loss(y_true.tolist(), y_pred.tolist())
        if isinstance(y_pred, list) and isinstance(y_true, list):
            return cross_entropy_loss(y_true, y_pred)
        if isinstance(y_pred, pd.DataFrame) and isinstance(y_true, pd.DataFrame):
            return cross_entropy_loss(y_true.values.tolist(), y_pred.values.tolist())
        else:
            raise ValueError(
                f"Invalid input types.\n y_pred: {type(y_pred)}\n y_true: {type(y_true)}"
            )

    def __repr__(self) -> str:
        """
        Get the string representation of the layer
        :return: str: The string representation of the layer
        """
        return f"Dense Layer: {self.weights.shape[0]} inputs, {self.weights.shape[1]} neurons"

    def __str__(self) -> str:
        """
        Get the string representation of the layer
        :return: str: The string representation of the layer
        """
        return f"Dense Layer: {self.weights.shape[0]} inputs, {self.weights.shape[1]} neurons"

