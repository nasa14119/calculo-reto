import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from grafos import get_graph, astar


def func(x, y):
    return (
        1000
        - 0.005 * x**2
        - 0.005 * y**2
        + 20 * np.sin(x * np.pi / 50) * np.cos(y * np.pi / 50)
    )


range_values = (0, 100)

x = np.linspace(-range_values[0], range_values[1], 100)
y = np.linspace(-range_values[0], range_values[1], 100)
start = [100, 50, func(100, 50)]
x, y = np.meshgrid(x, y)
z = func(x, y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x, y, z, cmap=cm.Blues, alpha=0.2)
max_index = np.argmax(z)

# Convertir el índice plano en índices de fila y columna
max_row, max_col = np.unravel_index(max_index, z.shape)

print(f"El índice de la fila con el valor máximo es: {max_row}")
print(f"El índice de la columna con el valor máximo es: {max_col}")

# Mostrar el punto máximo en el gráfico
puntos_max = [x[max_row, max_col], y[max_row, max_col], z[max_row, max_col]]
ax.scatter(puntos_max[0], puntos_max[1], puntos_max[2], color="red")
ax.scatter(start[0], start[1], start[2], color="blue")
ax.set_xlabel("x")
ax.set_ylabel("y")
cors = [0, 0]
# cors[0] = np.arange(np.float16(start[0]), np.float16(puntos_max[0]), -4)
# cors[1] = np.linspace(start[1], puntos_max[1], len(cors[0]))
# cors = np.transpose(cors)
# for cor in cors:
#     ax.scatter(cor[0], cor[1], func(cor[0], cor[1]), color="green")
print(f"({x[max_row, max_col]}, {y[max_row, max_col]}, {z[max_row, max_col]}")
data = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
graph = get_graph(data, start, puntos_max, ax)

# for u, v in graph.edges:
#     x = [u[0], v[0]]
#     y = [u[1], v[1]]
#     z = [u[2], v[2]]
#     ax.plot(x, y, z, c="gray", linewidth=0.5)
plt.show()
