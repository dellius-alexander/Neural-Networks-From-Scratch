from src.functions.activation import ActivationFunction
import numpy as np


def train_model(
        features: np.ndarray,
        labels: np.ndarray,
        activation: ActivationFunction,
        epochs: int,
        learning_rate: float):
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


