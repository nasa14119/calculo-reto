import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def func(x, y):
    return (
        1000
        - 0.005 * x**2
        - 0.005 * y**2
        + 20 * np.sin(x * np.pi / 50) * np.cos(y * np.pi / 50)
    )


range_values = (0, 100)

x = np.linspace(-range_values[0], range_values[1], 1000)
y = np.linspace(-range_values[0], range_values[1], 1000)
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
ax.scatter(x[max_row, max_col], y[max_row, max_col], z[max_row, max_col], color="red")
ax.scatter(start[0], start[1], start[2], color="blue")
# print(f"({x[max_row, max_col]}, {y[max_row, max_col]}, {z[max_row, max_col]}")
plt.show()
