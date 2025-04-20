import numpy as np


def func(x, y):
    return (
        1000
        - 0.005 * x**2
        - 0.005 * y**2
        + 20 * np.sin(x * np.pi / 50) * np.cos(y * np.pi / 50)
    )


def gradient_func(x, y):
    func_x = lambda x, y: -0.01 * x + np.pi * 2 * np.pow(5.0, -1) * np.cos(
        np.pi * x * np.pow(50.0, -1)
    ) * np.cos(np.pi * y * np.pow(50.0, -1))
    func_y = lambda x, y: -0.01 * y - np.pi * 2 * np.pow(5.0, -1) * np.sin(
        np.pi * x * np.pow(50.0, -1)
    ) * np.sin(np.pi * y * np.pow(50.0, -1))
    return np.array([func_x(x, y), func_y(x, y)])
