from exceptions.ImpossibleTreeException import ImpossibleTreeException
from util import union_find
class Graph_prim:
    def __init__(self, graph_matrix, k, MST_to_complete):
        self.uf = union_find.UnionFind(len(graph_matrix))
        self.graph_matrix = graph_matrix
        self.chromosome = []
        self.degree_limit = k
        self.edges_vault = []
        self.nodes_visited = {}
        self.MST_to_complete = self.check_and_TRANSFORM_MST(MST_to_complete)

    """
    Comprueba si el subarbol es correcto, es decir, no tiene ciclos, no sobrepasa el límite de grado y solo conecta
    nodos existentes. Y registra el grado de los nodos
    """
    def check_and_TRANSFORM_MST(self, MST_NC):
        biggest_node = 0
        for c, u, v in MST_NC:
            if u > biggest_node:
                biggest_node = u
            elif v > biggest_node:
                biggest_node = v
            if biggest_node >= len(self.graph_matrix):
                raise Exception("El grafo a completar contiene aristas que van a nodos inexistentes")
            degree_u = self.nodes_visited.get(u, 0)
            degree_v = self.nodes_visited.get(v, 0)

            if self.is_valid([c,u,v]):
                self.nodes_visited[u] = degree_u + 1
                self.nodes_visited[v] = degree_v + 1
            else:
                raise Exception("El grafo a completar excede la restricción de grado o contiene ciclos")
        return MST_NC

    """
    Sesga y guarda las aristas del grafo a partir de su matriz de adyacencia
    """
    def get_edges(self):
        edges = []

        for i in range(0, len(self.graph_matrix)):
            for j in range(0, len(self.graph_matrix)):
                if self.graph_matrix[i][j] != 0:
                    if i < j:
                        edges.append((self.graph_matrix[i][j] * self.chromosome[i] * self.chromosome[j], i, j))
        self.edges_vault = list(edges)
        return edges

    """
    Devuelve la arista más barata de las que conecta a los nodos visitados
    :param edges Aristas no usadas
        
    :return Arista
    """
    def get_cheapest_edge(self, edges):
        vertexs_filtered = list(filter(lambda edge: edge[1] in self.nodes_visited or edge[2] in self.nodes_visited, edges))
        vertexs_filtered.sort()
        if len(vertexs_filtered) <= 0:
            raise ImpossibleTreeException('No se puede completar el arbol')
        return vertexs_filtered[0]

    """
    Comprueba si una arista es válida

    :return False si añadirla violaría la restricción de grado o formaría ciclos
            True en caso contrario
    """
    def is_valid(self, edge, visited):
        if not self.check_degree_limit(edge):
            return False
        elif not self.uf.union(edge[1], edge[2]):
            return False
        else:
            return True

    """
    Comprueba si una arista viola la restriccion de grado

    :return False si añadirla violaría la restricción de grado
    """
    def check_degree_limit(self, edge):
        degree_u = self.nodes_visited.get(edge[1], 0)
        degree_v = self.nodes_visited.get(edge[2], 0)
        if (degree_u + 1) > self.degree_limit or (degree_v + 1) > self.degree_limit:
            return False
        return True

    def get_the_other_edge(self, edge):
        if edge[1] in self.nodes_visited:
            return edge[2], edge[1]
        else:
            return edge[1], edge[2]

    def prim(self, chromosome):
        self.chromosome = chromosome
        edges = self.get_edges()
        MST = []

        while len(self.nodes_visited) < (len(self.graph_matrix)):
            actual_edge = self.get_cheapest_edge(edges)
            edges.remove(actual_edge)
            if self.is_valid(actual_edge):
                MST.append(actual_edge)
                nodes = self.get_the_other_edge(actual_edge)
                self.nodes_visited[nodes[0]] = 1
                self.nodes_visited[nodes[1]] += 1
        return self.MST_to_complete + MST