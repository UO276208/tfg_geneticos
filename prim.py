class Graph:
    def __init__(self, graph_matrix, chromosome, k, input_data_type='gm'):
        self.graph_matrix = graph_matrix
        self.chromosome = chromosome
        self.degree_limit = k
        self.edges_vault = []
        self.input_data_type = input_data_type

    def get_edges(self):
        if self.input_data_type == 'gm':
            return self.get_edges_gm()

    def get_edges_gm(self):
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

    def get_cheapest_edge(self, vertex, edges):
        cheapest = edges[0]
        for edge in edges:
            if edge[1] == vertex:
                if edge[0] < cheapest[0]:
                    cheapest = edge
        return cheapest

    def is_valid(self, edge, MST, visited):
        if edge[1] in visited and edge[2] in visited:
            return False
        return self.check_degree_limit(edge, MST)

    def check_degree_limit(self, edge, MST):
        degree = 0
        for mst_edge in MST:
            if mst_edge[1] == edge[1] or mst_edge[2] == edge[1]:
                degree += 1
            if degree > (self.degree_limit - 1):
                return False
        return True

    def prim(self):
        start_vertex = self.chromosome[len(self.chromosome) - 1]
        edges = self.get_edges()
        edges.sort(reverse=True)
        MST = []
        nodes_visited = [start_vertex]
        actual_edge = self.get_cheapest_edge(start_vertex, edges)
        MST.append(actual_edge)
        nodes_visited.append(actual_edge[2])
        edges.remove(actual_edge)

        while len(nodes_visited) < (len(self.graph_matrix)):
            actual_edge = edges.pop()
            if self.is_valid(actual_edge, MST, nodes_visited):
                MST.append(actual_edge)
                nodes_visited.append(actual_edge[2])
        return MST

# graph_matrix = [[0,4,3,9],
#                 [4,0,8,10],
#                  [3,8,1,1],
#                   [9,10,1,0]]
#
# graph = Graph(graph_matrix,[3,4,5,1,0], 2)
# print(graph.prim(0))
# [(3, 0, 2), (1, 2, 3), (3, 0, 2)]
