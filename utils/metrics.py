import numpy as np

def mean_absolute_error(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def root_mean_squared_error(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def mean_absolute_percentage_error(actual, predicted):
    actual = np.where(actual == 0, 1e-10, actual)  # Avoid division by zero
    return np.mean(np.abs((actual - predicted) / actual)) * 100
