import numpy as np


def func(x, y):
    return (
        1000
        - 0.005 * x**2
        - 0.005 * y**2
        + 20 * np.sin(x * np.pi / 50) * np.cos(y * np.pi / 50)
    )
