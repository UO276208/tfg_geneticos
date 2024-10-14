class Graph_kruskal:
    def __init__(self, graph_matrix, k, input_data_type='gm'):
        self.graph_matrix = graph_matrix
        self.degree_limit = k
        self.edges_vault = []
        self.input_data_type = input_data_type

    def get_edges(self):
        if True:
            return self.get_edges_gm()

    def get_edges_gm(self):
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

    def kruskal(self):

        edges = self.get_edges()
        edges.sort(key=lambda x: x[0])
        MST = [edges.pop(0)]
        nodes_visited = [MST[0][1], MST[0][2]]
        while len(MST) < (len(self.graph_matrix[0])-1):
            actual_edge = edges.pop(0)
            if self.is_valid(actual_edge, MST, nodes_visited):
                MST.append(actual_edge)
                nodes_visited.append(actual_edge[2])

        return MST

graph_matrix = [[0,3,1,2],
                [3,0,2,4],
                [1,2,0,5],
                [2,4,5,0]]

graph = Graph_kruskal(graph_matrix,2, 2)
print(graph.kruskal())
