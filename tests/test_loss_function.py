#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from src.layer.dense import Dense
from src.utils.datasets import create_spiral_dataset
from src.functions.activation import Softmax, ReLU

class TestLossFunction(unittest.TestCase):

    def setUp(self):
        # Initialize activation functions
        self.softmax = Softmax()
        self.relu = ReLU()

        # Create a spiral dataset
        self.X, self.y = create_spiral_dataset(100, 3)
        self.y = np.array([self.y])

        # Create dense layers
        self.dense1 = Dense(2, 3)
        self.dense2 = Dense(3, 3)

    def tearDown(self):
        # Clean up any resources if needed
        pass

    def test_forward_pass_and_loss(self):
        print(f"Inputs: {self.X.shape}")
        print(f"Y is a spiral dataset: {self.y.shape}")

        # Forward pass through the first dense layer
        self.dense1.forward(self.X)
        print(f"Weights Layer 1: {self.dense1.weights.shape}")
        print(f"Biases Layer 1: {self.dense1.biases.shape}")
        print(f"Output Layer 1: {self.dense1.output.shape}")

        # Apply ReLU activation
        self.dense1.output = self.relu(self.dense1.output)

        # Forward pass through the second dense layer
        self.dense2.forward(self.dense1.output)
        print(f"Weights Layer 2: {self.dense2.weights.shape}")
        print(f"Biases Layer 2: {self.dense2.biases.shape}")
        print(f"Output Layer 2: {self.dense2.output.shape}")

        # Apply Softmax activation
        predictions = self.softmax(self.dense2.output)

        # Match the size of predictions to the size of y
        predictions = np.array([predictions[range(len(predictions)), self.y[0]]])
        print(f"Predictions: {predictions.shape}")
        print(f"True labels: {self.y.shape}")

        # Calculate the loss and print the results to 7 decimal places
        avg_loss, loss = self.dense2.loss(predictions, self.y)
        print(f"Loss: {avg_loss:.7f}")

        # Run ArgMax to get the predicted class
        predicted_class = np.argmax(predictions, axis=1)
        print(f"Predicted Class: {predicted_class.shape}")

if __name__ == '__main__':
    unittest.main()
