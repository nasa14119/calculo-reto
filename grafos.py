import heapq
from scipy.spatial import KDTree
import networkx as nx
import numpy as np

step = 1


def get_graph(puntos, start, end, ax):
    start = tuple(start)
    end = tuple(end)
    G = nx.Graph()
    # # Crear los nodos
    for p in puntos:
        G.add_node(tuple(p))
    G.add_node(start)
    G.add_node(end)
    # Usamos KDTree para encontrar vecinos cercanos
    tree = KDTree(puntos)

    for i, p in enumerate(puntos):
        # Buscar puntos cercanos a p dentro del radio
        indices = tree.query_ball_point(p, 4)
        for j in indices:
            if i != j:
                q = puntos[j]
                distancia = np.linalg.norm(
                    np.array(p) - np.array(q)
                )  # Calcular distancia euclidiana 3D
                G.add_edge(tuple(p), tuple(q), weight=distancia)  # Agregar la arista
    # G.add_edge(
    #     tuple(start), tuple(end), weight=np.linalg.norm(np.array(start) - np.array(end))
    # )

    # nodos = list(G.nodes)
    # ax.scatter(
    #     [nodo[0] for nodo in nodos],
    #     [nodo[1] for nodo in nodos],
    #     [nodo[2] for nodo in nodos],
    #     color="green",
    # )
    return G


def astar(graph, start, end):
    # Función heurística (distancia euclidiana)
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    # Inicialización de los valores
    open_list = []
    closed_list = set()
    came_from = {}

    # Diccionarios para almacenar los valores g y f
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    # Se utiliza una cola de prioridad basada en el valor de f(n)
    heapq.heappush(open_list, (f_score[start], start))

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

        # Recorremos los vecinos del nodo actual
        for neighbor in graph.neighbors(current):
            if neighbor in closed_list:
                continue

            tentative_g_score = g_score[current] + graph[current][neighbor]["weight"]

            # Si el vecino no está en open_list o encontramos un mejor camino
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # Si no se encuentra un camino
