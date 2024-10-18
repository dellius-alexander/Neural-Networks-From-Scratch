import numpy as np


def create_random_nonlinear_3D_dataset(x, y, z, n):
    # Generate random input data
    X_inputs = np.random.rand(x, n)
    Y_inputs = np.random.rand(y, n)
    Z_inputs = np.random.rand(z, n)
    # Apply a non-linear transformation (e.g., sine function)
    X = np.array(np.sin(X_inputs * np.pi))
    y = np.array(np.sin(Y_inputs * np.pi))
    z = np.array(np.sin(Z_inputs * np.pi))
    return X, y, z


def create_spiral_dataset(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, samples)
            + np.random.randn(samples) * 0.2
        )
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


# Sine sample dataset
def create_sin_dataset(samples=1000):
    X = np.arange(samples).reshape(-1, 1) / samples
    y = np.sin(2 * np.pi * X).reshape(-1, 1)
    return X, y


def create_vertical_data(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        X[ix] = np.c_[
            np.random.randn(samples) * 0.1 + class_number / 3,
            np.random.randn(samples) * 0.1 + 0.5,
        ]
        y[ix] = class_number
    return X, y
