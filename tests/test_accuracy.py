import time
from scipy.ndimage import zoom
import numpy as np
import unittest

from src.utils.datasets import create_vertical_data
from src.layer.dense import Dense
from src.functions.activation import Softmax, ReLU


# Test the accuracy of the model
class TestAccuracy(unittest.TestCase):

    # setup inputs and ground truth labels
    def setUp(self):
        self.start_time = time.time()
        # Create a vertical dataset
        self.inputs, self.y_true = create_vertical_data(100, 3)
        # Reshape the ground truth labels
        self.y_true = self.y_true.reshape(-1, 1)
        # Create a dense layer with 3 neurons with 2 inputs each
        self.dense = Dense(2, 3)
        # Create a dense layer with 3 neurons with 3 inputs each
        self.dense2 = Dense(3, 3)
        # Initialize activation function
        self.softmax = Softmax()
        self.relu = ReLU()
        # Set the number of epochs
        self.epochs = 10000
        print(f"Inputs: {self.inputs.shape}")
        print(f"Y is a vertical dataset: {self.y_true.shape}")
        print(f"Weights Layer 1: {self.dense.weights.shape}")
        print(f"Biases Layer 1: {self.dense.biases.shape}")
        print(f"Weights Layer 2: {self.dense2.weights.shape}")
        print(f"Biases Layer 2: {self.dense2.biases.shape}")

    def test_accuracy(self):
        # We will create some variables to track the best loss, accuracy and the associated weights and biases
        lowest_loss = 9999999
        best_accuracy = 0
        best_epoch = 0
        best_weights = self.dense.weights.copy()
        best_biases = self.dense.biases.copy()
        best_weights2 = self.dense2.weights.copy()
        best_biases2 = self.dense2.biases.copy()
        print("\n")

        # Iterate over the epochs
        for epoch in range(self.epochs):
            # Generate a new set of weights and biases for each iteration
            self.dense.weights = 0.5 * np.random.randn(2, 3)
            self.dense.biases = 0.1 * np.random.randn(1, 3)
            self.dense2.weights = 0.5 * np.random.randn(3, 3)
            self.dense2.biases = 0.1 * np.random.randn(1, 3)
            y_copy = self.y_true.copy()

            # Forward pass
            self.dense.forward(self.inputs)
            self.dense.activation = self.relu(self.dense.output)
            self.dense2.forward(self.dense.activation)
            predictions = self.softmax(self.dense2.output)

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
            avg_loss, loss = self.dense2.loss(predictions, y_copy)

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

            # Check if the loss is lower than the previous lowest loss
            if avg_loss < lowest_loss:
                print(f"Epoch: {epoch}, Next Lowest Loss: {avg_loss:.7f}, Next Highest Accuracy: {accuracy:.7f}")
                best_epoch = epoch
                lowest_loss = avg_loss
                best_accuracy = accuracy
                best_weights = self.dense.weights.copy()
                best_biases = self.dense.biases.copy()
                best_weights2 = self.dense2.weights.copy()
                best_biases2 = self.dense2.biases.copy()

            # Print the best weights and biases when the epoch is complete
            if epoch == self.epochs - 1:
                print(f"Total Epochs: {self.epochs}")
                print(f"Best Epoch: {best_epoch}, Best Loss: {lowest_loss:.7f}, Best Accuracy: {best_accuracy:.7f}")
                print(f"Best Weights Layer 1: \n{best_weights}")
                print(f"Best Biases Layer 1: \n{best_biases}")
                print(f"Best Weights Layer 2: \n{best_weights2}")
                print(f"Best Biases Layer 2: \n{best_biases2}")
                print(f"Duration: {time.time() - self.start_time}")


if __name__ == "__main__":
    unittest.main()
