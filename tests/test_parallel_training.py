#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

from src.functions.activation import Softmax, ReLU
from src.functions.train import parallel_training
from src.layer.dense import Dense
from src.utils.datasets import create_vertical_data
import numpy as np

# from src.utils.logger import getLogger
import unittest

# log = getLogger(__name__)


class TestAccuracyInParallel(unittest.TestCase):
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

    def test_accuracy_in_parallel(self):
        (
            best_epoch,
            lowest_loss,
            best_accuracy,
            best_weights,
            best_biases,
            best_weights2,
            best_biases2,
        ) = parallel_training(
            self.epochs,
            self.inputs,
            self.y_true,
            self.dense,
            self.dense2,
            self.relu,
            self.softmax,
        )

        self.assertIsNotNone(best_epoch)
        self.assertIsNotNone(lowest_loss)
        self.assertEqual(type(lowest_loss), np.float64)
        self.assertEqual(lowest_loss // 1, 36)
        self.assertIsNotNone(best_accuracy)
        self.assertIsNotNone(best_weights)
        self.assertIsNotNone(best_biases)
        self.assertIsNotNone(best_weights2)
        self.assertIsNotNone(best_biases2)

        print(
            f"""
        Best Epoch: {best_epoch}
        Lowest Loss: {lowest_loss}
        Best Accuracy: {best_accuracy}
        Best Weights: \n{best_weights}
        Best Biases: \n{best_biases}
        Best Weights2: \n{best_weights2}
        Best Biases2: \n{best_biases2}
        Duration: {time.time() - self.start_time}
        """
        )


if __name__ == "__main__":
    unittest.main()
