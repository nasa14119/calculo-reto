import numpy as np
from func import func

PRESICION = 10.0


def get_points_line(start, end):
    cors = [0, 0, 0]
    cors[0] = np.arange(np.float16(start[0]), np.float16(end[0]), -4)
    cors[1] = np.linspace(start[1], end[1], len(cors[0]))
    cors[2] = func(cors[0], cors[1])
    return np.transpose(cors)
