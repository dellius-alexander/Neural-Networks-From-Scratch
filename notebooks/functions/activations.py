from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
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
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x) -> np.ndarray:
        return np.array([1 for _ in x])


class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x) -> np.ndarray:
        return self.__call__(x) * (1 - self.__call__(x))


class SigmoidDerivative(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)

    def derivative(self, x) -> np.ndarray:
        return x * (1 - x)


class Tanh(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x) -> np.ndarray:
        return np.where(x <= 0, 0, 1)


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)


class Softmax(ActivationFunction):
    def __init__(self):
        self.last_output = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x))    # To avoid overflow
        self.last_output = exps / np.sum(exps, axis=0, keepdims=True)
        return self.last_output

    def derivative(self, x) -> np.ndarray:
        if self.last_output is None:
            raise ValueError("The softmax function has not been called yet.")
        return self.last_output * (1 - self.last_output)


# if __name__ == "__main__":
#     z = np.array([1, 2, 3])
#     probabilities = Softmax()
#     print(probabilities(z))
    # [0.09003057 0.24472847 0.66524096]
    # [0.09003057 0.24472847 0.66524096]