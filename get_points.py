import numpy as np
from func import func

PRESICION = 10


def get_points_line(start, end):
    cors = [0, 0, 0]
    cors[0] = np.arange(np.float16(start[0]), np.float16(end[0]), -2)
    cors[1] = np.linspace(start[1], end[1], len(cors[0]))
    cors[2] = func(cors[0], cors[1])
    return np.transpose(cors)


get_m = lambda cor1, cor2: (cor1[1] - cor2[1]) / (cor1[0] - cor2[0])


def get_points_graph(start, end, ax):
    cors = get_points_line(start, end)
    m = get_m(cors[0], cors[1])
    lin_func = (
        lambda x, i: -np.power(m, -1) * x + np.power(m, -1) * cors[i][0] + cors[i][1]
    )
    temp = list()
    for i in range(1, len(cors)):
        x1 = cors[i][0]
        x = np.linspace(x1 - 10, x1 + 10, PRESICION)
        y = lin_func(x, i)
        z = func(x, y)
        ax.scatter(x, y, z, c="green", alpha=0.4)
        temp.append(np.vstack([x, y, z]).T)
    cors = temp
    return cors
