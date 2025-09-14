import os
import tsplib95

def read_matrix(filename: str):
    # Ruta del archivo (como en tu código original)
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_archivo = os.path.join(ruta_actual, "..", filename)

    # Cargar el problema con tsplib95
    problem = tsplib95.load(ruta_archivo)

    # Lista real de nodos
    nodes = list(problem.get_nodes())
    n = len(nodes)

    # Crear matriz de adyacencia
    matriz = [[0 for _ in range(n)] for _ in range(n)]

    # Llenar matriz usando los nodos reales
    for i_idx, i in enumerate(nodes):
        for j_idx, j in enumerate(nodes):
            matriz[i_idx][j_idx] = problem.get_weight(i, j)

    return matriz
