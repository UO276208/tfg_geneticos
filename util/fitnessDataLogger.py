class FitnessDataLogger:
    def __init__(self, name):
        self.edges = None
        self.MST_list = []
        self.nodes_visited_list = []
        self.name = name
        self.filtered_edges_list = []
        self.data = ''
    def add_edges(self, edges):
        self.edges = '------All edges----------\n'
        for edge in edges:
            self.edges += '-' + str(edge) + '-\n'
    def add_filtered_edges(self, edges):
        edges_filtered = '------Edges filtered----------\n'
        for edge in edges:
            edges_filtered += '-' + str(edge) + '-\n'
        self.filtered_edges_list.append(edges_filtered)
    def add_nodes_visited(self, nodes_visited):
        self.nodes_visited_list.append(str(nodes_visited))
    def add_MST(self, MST):
        self.MST_list.append(MST)
