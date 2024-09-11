#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Annotated
from typing import Union, Tuple

import numpy as np
from numpy import ndarray

from src.utils.exception import RetryException, InvalidShapeException
from src.utils.logger import getLogger

log = getLogger(__name__)

Matrix = Annotated[Union[List, np.ndarray], "A matrix of values"]


def cross_entropy_loss(y_true: Matrix,
                       y_pred: Matrix) -> Tuple[any, Matrix]:
    """
    Calculate the cross-entropy loss between true labels and predicted probabilities.

    The cross-entropy loss is commonly used in classification problems to measure the
    difference between the predicted probabilities and the true labels.

    The cross-entropy loss is defined as:

    L = -1/N * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    where:
    - y_true is a binary vector of true labels.
    - y_pred is a vector of predicted probabilities.

    :param y_true: (Union[List[int], np.ndarray]): The true labels.
    Shape (N,n). Shape must be compatible and greater than 1.
    :param y_pred: (Union[List[float], np.ndarray]): The predicted probabilities.
    Shape (N,n). Shape must be compatible and greater than 1.
    :raises ValueError: If reshaping fails and the shapes remain incompatible.
    :return: (float): The average cross-entropy loss.
    """

    __loss__ = 0
    __losses__ = []

    try:

        # Convert lists to NumPy arrays for easier handling
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Replace and negative values with 0
        y_pred = np.where(y_pred < 0, 0, y_pred)

        # Replace any nan or negative values with 0
        y_pred = np.where(np.isnan(y_pred) | (y_pred < 0), 0, y_pred)

        # Prediction: Normalize the predicted probabilities but ensure the axis is 1 or make it 1
        y_pred = normalize_values(y_pred)

        # True labels: Ensure the axis is 1 or make it 1
        y_true = normalize_values(y_true)

        log.debug(f"Normalized Predicted Probabilities: \n{y_pred.shape}")
        log.debug(f"Normalized True Labels: \n{y_true.shape}")

        # Check if the shapes of the true labels and predicted probabilities are compatible.
        # If not, convert the shapes to be compatible, if possible.
        y_true, y_pred = convert_mismatch_shape(y_true, y_pred)

        # Compute the cross-entropy loss using vectorized operations.
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Compute the cross-entropy loss for each sample
        for p, q in zip(y_true, y_pred):
            __loss__ += -1 * (p * np.log(q))
            __losses__.append(__loss__)
        __loss__ = np.mean(__losses__)
    except Exception as e:
        log.error(e)
        raise e
    return __loss__, np.array(__losses__)


def sort_cross_entropy_loss(__predicted_probs: np.ndarray, __cross_entropy_losses: np.ndarray) -> tuple[
    np.ndarray, np.ndarray]:
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
    """Calculate the Mean Square Error (MSE) loss between the true and predicted values.

    The Mean Squared Error (MSE) is a widely used and straightforward loss
    function for regression tasks. It measures the average squared difference
    between the predicted values and the actual values in the dataset. The
    formula for MSE is as follows:

    .. math::
        \\text{J} &= (1/N) \\cdot \\sum_{i=1}^{N} L &= (1/N) \\cdot \\sum_{i=1}^{N} (\\hat{Y}_{i} - \\text{Y}_{i} )

    Where:

    - :math:`J`: is the cost function or the mean loss
    - :math:`N`: is the number of samples
    - :math:`L`: is the loss function or the cross-entropy loss
    - :math:`i`: is the index of the sample
    - :math:`\\sum`: is the sum function that returns the sum of the values in a matrix
    - :math:`\\hat{Y}_{i}`: is the predicted value for the :math:`i-th` sample
    - :math:`Y_{i}`: is the actual value for the :math:`i-th` sample


    :param y_true: (ndarray) - The true labels or values.
    :param y_pred: (ndarray) - The predicted labels or values.
    :returns: float: The mean square error loss.
    """
    # Calculate the mean of the squared differences
    __mse = np.mean((y_pred - y_true) ** 2)
    log.debug(__mse)
    return __mse


def check_for_shape_mismatch(y_true: ndarray, y_pred: ndarray) -> bool:
    """
    Check if the shapes of the true labels and predicted probabilities are compatible.

    :param y_true: (ndarray) - The true labels.
    :param y_pred: (ndarray) - The predicted probabilities.
    :returns: bool: True if the shapes are compatible, False otherwise.
    """
    return y_true.shape != y_pred.shape


def normalize_values(n_array: ndarray) -> ndarray:
    """
    Normalize the values in an array.

    :param n_array: (ndarray) - The array of values to normalize.
    :returns: ndarray: The normalized array of values.
    """
    return n_array / np.linalg.norm(n_array)


def convert_mismatch_shape(y_true: ndarray, y_pred: ndarray, **kwargs) -> Tuple[ndarray, ndarray]:
    """
    Convert the shapes of the true labels and predicted probabilities to be compatible.

    :param y_true: (ndarray) - The true labels.
    :param y_pred: (ndarray) - The predicted probabilities.
    :returns: Tuple[ndarray, ndarray]: The true labels and predicted probabilities with compatible shapes.
    """

    __error_counter = kwargs.get("error_counter", 0)
    __max_retries = kwargs.get("max_retries", 6)
    __mismatch_type = kwargs.get("mismatch_type", {
        "transpose": {'success': False, 'shape': None, 'message': 'None'},
        "reshape": {'success': False, 'shape': None, 'message': 'None'},
        "concatenate": {'success': False, 'shape': None, 'message': 'None'},
        "broadcast": {'success': False, 'shape': None, 'message': 'None'},
        "expand_dims": {'success': False, 'shape': None, 'message': 'None'},
        "squeeze": {'success': False, 'shape': None, 'message': 'None'},
    })
    __inputs = {
        "y_true": y_true,
        "y_pred": y_pred
    }

    try:
        # Ensure the predicted probabilities are valid (in the range [0, 1])
        if np.any(y_pred < 0) or np.any(y_pred > 1):
            raise ValueError(f"""Predicted probabilities must be between 0 and 1.
                                \nPredicted probabilities: \n{y_pred.shape}
                                \nTrue labels: \n{y_true.shape}""")

        # Attempt to reshape if the shapes don't match
        if y_true.shape != y_pred.shape and __mismatch_type["reshape"]['message'] == 'None':
            try:
                # Reshape the true labels by inverting the shape to possibly match the predicted probabilities
                y_true = np.reshape(y_true, (y_pred.shape[1], y_pred.shape[0]))
                log.debug(f"Reshaped True Labels: \n{y_true.shape}")

                # Check if the shapes are compatible again
                if y_true.shape != y_pred.shape:
                    __mismatch_type["reshape"] = {'success': False, 'shape': y_true.shape, 'message': 'Reshape failed'}
                    raise InvalidShapeException(f"""Shapes of true labels and predicted probabilities must be compatible.
                    \nTrue Labels Shape: {y_true.shape}
                    \nPredicted Probabilities Shape: {y_pred.shape}""")
            except ValueError as e:
                log.error(e)
                raise e

        # Attempt to transpose if the shapes don't match
        if y_true.shape != y_pred.shape and __mismatch_type["transpose"]['message'] == 'None':
            try:
                # Transpose the true labels to match the predicted probabilities
                y_true = np.transpose(y_true)
                log.debug(f"Transposed True Labels: \n{y_true.shape}")

                # Check if the shapes are compatible again
                if y_true.shape != y_pred.shape:
                    __mismatch_type["transpose"] = {'success': False, 'shape': y_true.shape, 'message': 'Transpose failed'}
                    raise InvalidShapeException(f"""Shapes of true labels and predicted probabilities must be compatible.
                    \nTrue Labels Shape: {y_true.shape}
                    \nPredicted Probabilities Shape: {y_pred.shape}""")
            except ValueError as e:
                log.error(e)
                raise e

        # Attempt to concatenate if the shapes don't match
        if y_true.shape != y_pred.shape and __mismatch_type["concatenate"]['message'] == 'None':
            try:
                # Concatenate the true labels to match the predicted probabilities
                y_true = np.concatenate(y_true, axis=0)
                log.debug(f"Concatenated True Labels: \n{y_true.shape}")

                # Check if the shapes are compatible again
                if y_true.shape != y_pred.shape:
                    __mismatch_type["concatenate"] = {'success': False, 'shape': y_true.shape,
                                                      'message': 'Concatenate failed'}
                    raise InvalidShapeException(f"""Shapes of true labels and predicted probabilities must be compatible.
                    \nTrue Labels Shape: {y_true.shape}
                    \nPredicted Probabilities Shape: {y_pred.shape}""")
            except ValueError as e:
                log.error(e)
                raise e

        # Attempt to broadcast if the shapes don't match
        if y_true.shape != y_pred.shape and __mismatch_type["broadcast"]['message'] == 'None':
            try:
                # Broadcast the true labels to match the predicted probabilities
                y_true = np.broadcast(y_true, y_pred)
                log.debug(f"Broadcasted True Labels: \n{y_true.shape}")

                # Check if the shapes are compatible again
                if y_true.shape != y_pred.shape:
                    __mismatch_type["broadcast"] = {'success': False, 'shape': y_true.shape, 'message': 'Broadcast failed'}
                    raise InvalidShapeException(f"""Shapes of true labels and predicted probabilities must be compatible.
                    \nTrue Labels Shape: {y_true.shape}
                    \nPredicted Probabilities Shape: {y_pred.shape}""")
            except ValueError as e:
                log.error(e)
                raise e

        # Attempt to expand dimensions if the shapes don't match
        if y_true.shape != y_pred.shape and __mismatch_type["expand_dims"]['message'] == 'None':
            try:
                # Expand dimensions of the true labels to match the predicted probabilities
                y_true = np.expand_dims(y_true, axis=0)
                log.debug(f"Expanded True Labels: \n{y_true.shape}")

                # Check if the shapes are compatible again
                if y_true.shape != y_pred.shape:
                    __mismatch_type["expand_dims"] = {'success': False, 'shape': y_true.shape,
                                                      'message': 'Expand dims failed'}
                    raise InvalidShapeException(f"""Shapes of true labels and predicted probabilities must be compatible.
                    \nTrue Labels Shape: {y_true.shape}
                    \nPredicted Probabilities Shape: {y_pred.shape}""")
            except ValueError as e:
                log.error(e)
                raise e

        # Attempt to squeeze if the shapes don't match
        if y_true.shape != y_pred.shape and __mismatch_type["squeeze"]['message'] == 'None':
            try:
                # Squeeze the true labels to match the predicted probabilities
                y_true = np.squeeze(y_true)
                log.debug(f"Squeezed True Labels: \n{y_true.shape}")

                # Check if the shapes are compatible again
                if y_true.shape != y_pred.shape:
                    __mismatch_type["squeeze"] = {'success': False, 'shape': y_true.shape, 'message': 'Squeeze failed'}
                    raise InvalidShapeException(f"""Shapes of true labels and predicted probabilities must be compatible.
                    \nTrue Labels Shape: {y_true.shape}
                    \nPredicted Probabilities Shape: {y_pred.shape}""")
            except ValueError as e:
                log.error(e)
                raise e
    except Exception as e:
        # Retry recursively until the maximum number of retries is reached
        if __error_counter < __max_retries:
            log.error(f"Error: {e}")
            log.info(f"Retrying... Attempt {__error_counter + 1}")
            return convert_mismatch_shape(__inputs['y_true'], __inputs['y_pred'], error_counter=__error_counter + 1, max_retries=__max_retries)
        else:
            raise RetryException(f"Maximum number of retries reached. \nError: {e}")

    return y_true, y_pred


if __name__ == "__main__":
    # Now we create a dense layer with 3 neurons with 2 inputs each and 2 dense layers; the first layer has 3 neurons with 2 inputs each and the second layer has 3 neurons with 3 inputs each.
    from src.layer.dense import Dense
    from src.utils.datasets import create_spiral_dataset
    from src.functions.activation import Softmax, ReLU

    # Initialize activation function
    softmax = Softmax()
    relu = ReLU()

    # Create a spiral dataset
    X, y = create_spiral_dataset(100, 3)
    y = np.array([y])
    print(f"Inputs: {X.shape}")
    print(f"Y is a spiral dataset: {y.shape}")

    # Create a dense layer with 3 neurons with 2 inputs each
    dense = Dense(2, 3)

    # Lets do the forward pass
    dense.forward(X)
    print(f"Weights Layer 1: {dense.weights.shape}")
    print(f"Biases Layer 1: {dense.biases.shape}")
    print(f"Output Layer 1: {dense.output.shape}")

    # Run the activation function ReLU
    dense.output = relu(dense.output)

    # Create a dense layer with 3 neurons with 3 inputs each
    dense2 = Dense(3, 3)

    # Lets do the forward pass
    dense2.forward(dense.output)
    print(f"Weights Layer 2: {dense2.weights.shape}")
    print(f"Biases Layer 2: {dense2.biases.shape}")
    print(f"Output Layer 2: {dense2.output.shape}")

    # TODO: These final outputs are also our “confidence scores.” The higher the confidence score, the
    #  more confident the model is that the input belongs to that class.

    # Run the activation function ReLU
    predictions = softmax(dense2.output)

    # Match the size of predictions to the size of y
    predictions = np.array([predictions[range(len(predictions)), y[0]]])
    print(f"Predictions: {predictions.shape}")
    print(f"Predictions Data: \n{predictions[:5, 0]}")
    print(f"Predictions: {predictions.shape}")
    print(f"True labels: {y.shape}")
    print(f"Predictions Data: \n{predictions[:5, 0]}")

    # Calculate the loss and print the results to 7 decimal places
    avg_loss, loss = dense2.loss(predictions, y.T)
    print(f"Loss: {avg_loss:.7f}")  # Loss: 5.7037784

    # Run ArgMax to get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    print(f"Predicted Class: {predicted_class.shape}")
