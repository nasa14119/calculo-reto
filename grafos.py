import heapq
import networkx as nx
import numpy as np
from func import gradient_func
from utils import loader
from tqdm import tqdm

step = 1
get_distance = lambda cor1, cor2: np.linalg.norm([cor2[0] - cor1[0], cor2[1] - cor1[1]])


def get_graph(puntos, start, end, ax):
    start = tuple(start)
    end = tuple(end)
    G = nx.Graph()
    # # Crear los nodos
    G.add_node(start)
    G.add_node(end)
    for line in loader(puntos, desc="creando todos los nodos"):
        for punto in line:
            G.add_node(tuple(punto))

    # Adding first edge
    for punto in loader(puntos[-1], desc="conexiones con el nodo base"):
        distance = get_distance(end, punto)
        if distance <= 5:
            G.add_edge(tuple(punto), end, weight=distance)

    # Adding first edges
    for punto in loader(puntos[0], desc="conexiones con el nodo final o destino"):
        distance = get_distance(punto, start)
        if distance <= 5:
            G.add_edge(tuple(punto), start, weight=distance)
    for i in loader(range(0, len(puntos) - 1), desc="generando los posibles caminos"):
        line = puntos[i]
        next_line = puntos[i + 1]
        for punto in line:
            for next_node in next_line:
                distance = get_distance(next_node, punto)
                if distance < 5:
                    G.add_edge(tuple(punto), tuple(next_node), weight=distance)
    # for u, v in G.edges:
    #     x = [u[0], v[0]]
    #     y = [u[1], v[1]]
    #     z = [u[2], v[2]]
    #     ax.plot(x, y, z, c="gray", linewidth=2.5, alpha=0.5)
    return G


def astar(graph, start, end):
    start = tuple(start)
    end = tuple(end)
    traversed_paths = nx.Graph()

    # Función heurística (distancia euclidiana)
    def heuristic(a, c):
        V = np.array(a) - np.array(c)
        u = np.array([V[0], V[1]]) / np.linalg.norm(V)
        grad_punto = gradient_func(a[0], a[1])
        ang = np.abs(np.degrees(np.arctan(np.dot(grad_punto, u))))
        if isinstance(ang, np.ndarray):
            return 1
        return np.exp(ang - 31)

    # Inicialización de los valores
    open_list = []
    closed_list = set()
    came_from = {}

    # Diccionarios para almacenar los valores g y f
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    # Se utiliza una cola de prioridad basada en el valor de f(n)
    heapq.heappush(open_list, (f_score[start], start))
    with tqdm(total=len(graph.nodes), desc="A-star traversing") as bar:
        while open_list:
            _, current = heapq.heappop(open_list)
            # Si hemos llegado al final
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            closed_list.add(current)
            bar.update()
            # Recorremos los vecinos del nodo actual
            for neighbor in graph.neighbors(current):
                if neighbor in closed_list:
                    continue

                tentative_g_score = (
                    g_score[current] + graph[current][neighbor]["weight"]
                )

                # Si el vecino no está en open_list o encontramos un mejor camino
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, current)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
        return None  # Si no se encuentra un camino
