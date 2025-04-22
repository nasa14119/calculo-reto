import sys
import os

# Añade el directorio padre al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import main
import numpy as np
from func import get_angulo

file = open("./dist.md", "w")
path, _ = main()


def get_row(i):
    get_distance = lambda cor1, cor2: np.linalg.norm(
        [cor2[0] - cor1[0], cor2[1] - cor1[1]]
    )
    node, next_node = path[i], path[i + 1]
    x, y, z = node
    distance_to = get_distance(node, next_node)
    angulo = get_angulo(node, next_node)
    line = f"|{i+1}|({x:.2f}, {y:.2f}, {z:.2f})|{angulo:0.2f}°|{distance_to:.2f}|\n"
    file.write(line)
    return distance_to, angulo


def build():
    file.write("# Resultados de la simulación\n")
    file.write("|Indice|Coordenada|Ángulo|Distancia siguiente punto|\n")
    file.write("|:----:|:-------:|:----:|:------:|\n")
    total_distance = 0
    for i in range(len(path) - 1):
        distance_current, _ = get_row(i)
        total_distance += distance_current
    file.write("\n")
    file.write(f"Distancia total recorrida: {total_distance}\n")
    file.write(
        f"<br>Punto Maximo:({path[-1][0]:0.2f},{path[-1][1]:0.2f},{path[-1][2]:0.2f})\n"
    )

    file.close()


if __name__ == "__main__":
    build()
