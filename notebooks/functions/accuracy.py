import numpy as np


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the accuracy of the model.
    One of the fundamental metrics for classification problems, accuracy refers to the measure of correct predictions made by the model. It is calculated as the number of correct predictions divided by all predictions. The accuracy formula in machine learning is as follows:

    - Correct Predictions: The number of correct predictions are the sum of all predictions where the predicted value is equal to the actual value.
    - Accuracy = (Number of correct predictions) / (Total number of predictions)

    :param y_true: np.ndarray: The true labels of the dataset
    :param y_pred: np.ndarray: The predicted labels of the dataset
    :return: float: The accuracy of the model
    """
    # Ensure the inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute the number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)

    # Calculate the accuracy
    accuracy = correct_predictions / len(y_true)

    return accuracy
