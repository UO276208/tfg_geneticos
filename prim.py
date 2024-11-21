class Graph_prim:
    def __init__(self, graph_matrix, chromosome, k, input_data_type='gm'):
        self.graph_matrix = graph_matrix
        self.chromosome = chromosome
        self.degree_limit = k
        self.edges_vault = []

    def get_edges(self):
        edges = []

        for i in range(0, len(self.graph_matrix)):
            for j in range(0, len(self.graph_matrix)):
                if i != j:
                    if i < j:
                        edges.append((self.graph_matrix[i][j] * self.chromosome[i] * self.chromosome[j], i, j))
                    else:
                        edges.append((self.graph_matrix[i][j] * self.chromosome[i] * self.chromosome[j], j, i))
        self.edges_vault = list(edges)
        return edges

    def get_cheapest_edge(self, vertexs, edges):
        vertexs_filtered = list(filter(lambda edge: edge[1] in vertexs or edge[2] in vertexs, edges))
        vertexs_filtered.sort()
        return vertexs_filtered[0]

    def is_valid(self, edge, visited):
        if edge[1] in visited and edge[2] in visited:
            return False
        return self.check_degree_limit(edge, visited)

    def check_degree_limit(self, edge, visited):
        degree_u = visited.get(edge[1], 0)
        degree_v = visited.get(edge[2], 0)
        if (degree_u + 1) > self.degree_limit or (degree_v + 1) > self.degree_limit:
            return False
        return True

    def get_the_other_edge(self, nodes_visited, edge):
        if edge[1] in nodes_visited:
            return edge[2], edge[1]
        else:
            return edge[1], edge[2]

    def prim(self):
        start_vertex = self.chromosome[len(self.chromosome) - 1]
        edges = self.get_edges()
        MST = []
        nodes_visited = {start_vertex: 1}
        actual_edge = self.get_cheapest_edge([start_vertex], edges)
        MST.append(actual_edge)
        nodes_visited[self.get_the_other_edge(nodes_visited, actual_edge)[0]] = 1
        edges.remove(actual_edge)

        while len(nodes_visited) < (len(self.graph_matrix)):
            actual_edge = self.get_cheapest_edge(nodes_visited, edges)
            edges.remove(actual_edge)
            if self.is_valid(actual_edge, nodes_visited):
                MST.append(actual_edge)
                nodes = self.get_the_other_edge(nodes_visited, actual_edge)
                nodes_visited[nodes[0]] = 1
                nodes_visited[nodes[1]] += 1
        return MST

# graph_matrix = [[0,4,3,9],
#                 [4,0,8,10],
#                  [3,8,1,1],
#                   [9,10,1,0]]
#
# graph = Graph(graph_matrix,[3,4,5,1,0], 2)
# print(graph.prim(0))
# [(3, 0, 2), (1, 2, 3), (3, 0, 2)]
