from util import union_find
class Graph_kruskal:
    def __init__(self, graph_matrix, chromosome, k, input_data_type='gm'):
        self.uf = union_find.UnionFind(len(graph_matrix))
        self.graph_matrix = graph_matrix
        self.chromosome = chromosome
        self.degree_limit = k
        self.edges_vault = []
        self.input_data_type = input_data_type

    def get_edges(self):
        edges = []

        for i in range(0, len(self.graph_matrix)):
            for j in range(0, len(self.graph_matrix)):
                if self.graph_matrix[i][j] != 0:
                    if i < j:
                        edges.append((self.graph_matrix[i][j] * self.chromosome[i] * self.chromosome[j], i, j))
        self.edges_vault = list(edges)
        return edges

    def is_valid(self, edge, visited):
        if not self.check_degree_limit(edge, visited):
            return False
        elif not self.uf.union(edge[1], edge[2]):
            return False
        else:
            return True

    def check_degree_limit(self, edge, visited):
        degree_u = visited.get(edge[1], 0)
        degree_v = visited.get(edge[2], 0)
        if (degree_u + 1) > self.degree_limit or (degree_v + 1) > self.degree_limit:
            return False
        return True

    def kruskal(self):
        edges = self.get_edges()
        edges.sort(key=lambda x: x[0])
        actual_edge = edges.pop()
        MST = [actual_edge]
        self.uf.union(actual_edge[1], actual_edge[2])
        nodes_visited = {MST[0][1]: 1, MST[0][2]: 1}
        while len(nodes_visited) < (len(self.graph_matrix[0])):
            actual_edge = edges.pop()
            if self.is_valid(actual_edge, nodes_visited):
                MST.append(actual_edge)
                #Saco los nodos que toca el arco añadido para actualizar su grados en el diccionario nodes_visited
                u, v = actual_edge[1], actual_edge[2]
                nodes_visited[u] = nodes_visited.get(u, 0) + 1
                nodes_visited[v] = nodes_visited.get(v, 0) + 1

        return MST