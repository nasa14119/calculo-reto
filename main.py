import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from grafos import get_graph, astar
from get_points import get_points_graph, get_points_line
from func import func


x = np.linspace(10, 100, 100)
y = np.linspace(-20, 60, 100)
start = [100, 50, func(100, 50)]
x, y = np.meshgrid(x, y)
z = func(x, y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x, y, z, cmap=cm.Blues, alpha=0.2)
max_index = np.argmax(z)

# Convertir el índice plano en índices de fila y columna
max_row, max_col = np.unravel_index(max_index, z.shape)

# Mostrar el punto máximo en el gráfico
puntos_max = [x[max_row, max_col], y[max_row, max_col], z[max_row, max_col]]
ax.scatter(puntos_max[0], puntos_max[1], puntos_max[2], color="red")
ax.scatter(start[0], start[1], start[2], color="blue")
ax.set_xlabel("x")
ax.set_ylabel("y")
# cors = get_points_line(start, puntos_max, ax)
data = get_points_graph(start, puntos_max, ax)
# data = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
graph = get_graph(data, start, puntos_max, ax)
path = astar(graph, start, puntos_max)
for i in range(len(path) - 1):
    p1 = path[i]
    p2 = path[i + 1]
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    z = [p1[2], p2[2]]
    ax.plot(x, y, z, c="red", linewidth=2, label="Camino óptimo" if i == 0 else "")
# for u, v in graph.edges:
#     x = [u[0], v[0]]
#     y = [u[1], v[1]]
#     z = [u[2], v[2]]
#     ax.plot(x, y, z, c="gray", linewidth=0.5)
plt.show()
