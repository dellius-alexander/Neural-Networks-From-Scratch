#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
activation.py
==============

This module contains implementations of various activation functions commonly used in neural networks and deep learning. Each activation function is implemented as a class that inherits from an abstract base class `ActivationFunction`, which defines the interface for activation functions, including the methods for computing the activation and its derivative.

Activation Functions Implemented:
---------------------------------
1. **Linear**: Implements the identity activation function, which returns the input as is.
2. **Sigmoid**: Implements the sigmoid activation function, which outputs values between 0 and 1.
3. **Tanh**: Implements the hyperbolic tangent activation function, which outputs values between -1 and 1.
4. **ReLU**: Implements the Rectified Linear Unit (ReLU) activation function, which outputs the input if it is positive; otherwise, it outputs zero.
5. **LeakyReLU**: Implements the Leaky ReLU activation function, which allows a small, non-zero gradient when the input is negative.
6. **Softmax**: Implements the softmax activation function, which outputs a probability distribution over classes.

Each class provides:
- `__call__(x: np.ndarray) -> np.ndarray`: Method to compute the activation function.
- `derivative(x: np.ndarray) -> np.ndarray`: Method to compute the derivative of the activation function.

Dependencies:
-------------
- numpy: The module relies on numpy for efficient numerical computations.

"""
from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    """
    Abstract base class for all activation functions.

    Methods:
        - __call__(x: np.ndarray) -> np.ndarray: Computes the activation function for input x.
        - derivative(x: np.ndarray) -> np.ndarray: Computes the derivative of the activation function.
    """
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """The activation function.

        :param x: np.ndarray: The input to the activation function.
        :return: np.ndarray: The output of the activation function.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, x) -> np.ndarray:
        """The derivative of the activation function.

        :param x: np.ndarray: The input to the activation function.
        :return: np.ndarray: The derivative of the activation function.
        """
        raise NotImplementedError


class Linear(ActivationFunction):
    """
    Implements the Linear (identity) activation function.

    Methods:
        - __call__(x: np.ndarray) -> np.ndarray: Returns the input as is.
        - derivative(x: np.ndarray) -> np.ndarray: Returns an array of ones,
        since the derivative of a linear function is 1.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Linear (identity) activation function.

        :param x: (np.ndarray) Input array of any shape.
        :return: (np.ndarray) The input array is returned as is.
        """
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the Linear (identity) activation function.

        :param x: (np.ndarray) Input array of any shape.
        :return: (np.ndarray) An array of ones with the same shape as the input,
                 since the derivative of a linear function is 1.
        """
        return np.array([1 for _ in x])


class Sigmoid(ActivationFunction):
    """
    Implements the Sigmoid activation function.

    Mathematically, the Sigmoid function is defined as:

    :math: σ(x) = 1/(1 + e^{-x})

    Calculating the Sigmoid function for a given input array `inputs` can be done as follows:

        >>> # Get unnormalized probabilities
        >>> exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)
        >>> # Normalize them for each sample
        >>> probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    Methods:
        - __call__(x: np.ndarray) -> np.ndarray: Computes the Sigmoid activation function: :math:1/(1 + e^{-x}).
        - derivative(x: np.ndarray) -> np.ndarray: Computes the derivative of the Sigmoid function.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Sigmoid activation function.

        .. math::
            σ(x) = 1/(1 + e^{-x})

        :param x: (np.ndarray) Input array of any shape.
        :return: (np.ndarray) The Sigmoid function applied to each element of the input.
                 The output values are in the range (0, 1).
        """
        if isinstance(x, list):
            x = np.array(x)
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the Sigmoid activation function.

        :param x: (np.ndarray) Input array of any shape, typically the output of the Sigmoid function.
        :return: (np.ndarray) The derivative of the Sigmoid function, given by
                 sigmoid(x) * (1 - sigmoid(x)).
        """
        return self.__call__(x) * (1 - self.__call__(x))


class Tanh(ActivationFunction):
    """
    Implements the Tanh (Hyperbolic Tangent) activation function.

    Methods:
        - __call__(x: np.ndarray) -> np.ndarray: Computes the Tanh function: tanh(x).
        - derivative(x: np.ndarray) -> np.ndarray: Computes the derivative of the Tanh function.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Tanh (Hyperbolic Tangent) activation function.

        :param x: (np.ndarray) Input array of any shape.
        :return: (np.ndarray) The Tanh function applied to each element of the input.
                 The output values are in the range (-1, 1).
        """
        return (np.tanh(x) - 1) / (np.tanh(x) + 1)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the Tanh activation function.

        :param x: (np.ndarray) Input array of any shape, typically the output of the Tanh function.
        :return: (np.ndarray) The derivative of the Tanh function, given by
                 1 - tanh(x)^2.
        """
        return 1 - np.tanh(x) ** 2


class ReLU(ActivationFunction):
    """
    Implements the Rectified Linear Unit (ReLU) activation function.

    Methods:
        - __call__(x: np.ndarray) -> np.ndarray: Returns the input if positive; otherwise, returns zero.
        - derivative(x: np.ndarray) -> np.ndarray: Returns 1 if the input is positive; otherwise, returns 0.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the ReLU (Rectified Linear Unit) activation function.

        :param x: (np.ndarray) Input array of any shape.
        :return: (np.ndarray) The ReLU function applied to each element of the input.
                 The output is the input if it is positive; otherwise, it is zero.
        """
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the ReLU activation function.

        :param x: (np.ndarray) Input array of any shape.
        :return: (np.ndarray) The derivative of the ReLU function, which is 1 for
                 positive input values and 0 for negative input values.
        """
        return np.where(x <= 0, 0, 1)


class LeakyReLU(ActivationFunction):
    """
    Implements the Leaky ReLU activation function.

    Attributes:
        - alpha (float): Slope for the negative input range. Default is 0.01.

    Methods:
        - __call__(x: np.ndarray) -> np.ndarray: Returns the input if positive; otherwise, returns alpha * input.
        - derivative(x: np.ndarray) -> np.ndarray: Returns 1 if the input is positive; otherwise, returns alpha.
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Computes the Leaky ReLU activation function.

        :param x: (np.ndarray) Input array of any shape.
        :param alpha: (float) Slope for the negative input range (default is 0.01).
        :return: (np.ndarray) The Leaky ReLU function applied to each element of the input.
                 The output is the input if it is positive; otherwise, it is alpha * input.
        """
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Computes the derivative of the Leaky ReLU activation function.

        :param x: (np.ndarray) Input array of any shape.
        :param alpha: (float) Slope for the negative input range (default is 0.01).
        :return: (np.ndarray) The derivative of the Leaky ReLU function, which is 1 for
                 positive input values and alpha for negative input values.
        """

        return np.where(x > 0, 1, self.alpha)


class Softmax(ActivationFunction):
    """
    Implements the Softmax activation function.

    Methods:
        - __call__(x: np.ndarray) -> np.ndarray: Computes the softmax function over the input array.
        - derivative(x: np.ndarray) -> np.ndarray: Computes the derivative of the softmax function.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Softmax activation function.

        :param x: (np.ndarray) Input array of any shape.
        :return: (np.ndarray) The Softmax function applied to the input. Each element is
                 transformed to a probability, and the sum of all elements is 1.
        """
        exps = np.exp(x - np.max(x))    # To avoid overflow
        return exps / np.sum(exps, axis=0, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the Softmax activation function.

        :param x: (np.ndarray) Input array of any shape, typically the output of the Softmax function.
        :return: (np.ndarray) The derivative of the Softmax function.
        """
        try:
            if len(x) <= 1:
                raise ValueError("The ndarray seems to empty. "
                                 "Softmax derivative can not be run on empty array.")
        except ValueError as e:
            print(e)
        finally:
            return x * (1 - x)


# if __name__ == "__main__":
#     z = np.array([1, 2, 3])
#     probabilities = Softmax()
#     print(probabilities(z))
    # [0.09003057 0.24472847 0.66524096]
    # [0.09003057 0.24472847 0.66524096]