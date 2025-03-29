from exceptions.ImpossibleTreeException import ImpossibleTreeException
from util import union_find
class Graph_prim:
    def __init__(self, graph_matrix, chromosome, k, input_data_type='gm'):
        self.uf = union_find.UnionFind(len(graph_matrix))
        self.graph_matrix = graph_matrix
        self.chromosome = chromosome
        self.degree_limit = k
        self.edges_vault = []
        self.input_data_type = input_data_type
        self.n_violations = 0

    def get_edges(self):
        edges = []

        for i in range(0, len(self.graph_matrix)):
            for j in range(0, len(self.graph_matrix)):
                if i != j:
                    if i < j:
                        edges.append((self.graph_matrix[i][j] * self.chromosome[i] * self.chromosome[j], i, j))
        self.edges_vault = list(edges)
        return edges

    def get_edges_not_biased(self):
        edges = []

        for i in range(0, len(self.graph_matrix)):
            for j in range(0, len(self.graph_matrix)):
                if i != j:
                    if i < j:
                        edges.append((self.graph_matrix[i][j], i, j))
                    else:
                        edges.append((self.graph_matrix[i][j], j, i))
        self.edges_vault = list(edges)
        return edges

    def get_cheapest_edge(self, vertexs, edges):
        vertexs_filtered = list(filter(lambda edge: edge[1] in vertexs or edge[2] in vertexs, edges))
        vertexs_filtered.sort()
        if len(vertexs_filtered) <= 0:
            raise ImpossibleTreeException('No se puede completar el arbol')
        return vertexs_filtered[0]

    def is_valid(self, edge, visited):
        if not self.uf.union(edge[1], edge[2]):
            return False
        else:
            self.check_degree_limit(edge, visited)
            return True

    def check_degree_limit(self, edge, visited):
        degree_u = visited.get(edge[1], 0)
        degree_v = visited.get(edge[2], 0)
        if (degree_u + 1) > self.degree_limit:
            self.n_violations += 1
        if (degree_v + 1) > self.degree_limit:
            self.n_violations += 1

    def get_the_other_edge(self, nodes_visited, edge):
        if edge[1] in nodes_visited:
            return edge[2], edge[1]
        else:
            return edge[1], edge[2]

    def prim(self):
        start_vertex = self.chromosome[len(self.chromosome) - 1]
        edges = self.get_edges()
        MST = []
        actual_edge = self.get_cheapest_edge([start_vertex], edges)
        nodes_visited = {actual_edge[1]: 1, actual_edge[2]: 1}
        MST.append(actual_edge)
        self.uf.union(actual_edge[1], actual_edge[2])
        edges.remove(actual_edge)

        while len(nodes_visited) < (len(self.graph_matrix)):
            actual_edge = self.get_cheapest_edge(nodes_visited, edges)
            edges.remove(actual_edge)
            if self.is_valid(actual_edge, nodes_visited):
                MST.append(actual_edge)
                nodes = self.get_the_other_edge(nodes_visited, actual_edge)
                nodes_visited[nodes[0]] = 1
                nodes_visited[nodes[1]] += 1

        return MST, self.n_violations
