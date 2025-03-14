class FitnessDataLogger:
    def __init__(self):
        self.edges = None
        self.MST_list = []
        self.nodes_visited_list = []
        self.filtered_edges_list = []
        self.data = ''
    def add_edges(self, edges):
        self.edges = '------All edges----------\n'
        for edge in edges:
            self.edges += '-' + str(edge) + '-\n'
    def add_filtered_edges(self, edges):
        edges_filtered = ''
        for edge in edges:
            edges_filtered += '-' + str(edge) + '-\n'
        self.filtered_edges_list.append(edges_filtered)
    def add_nodes_visited(self, nodes_visited):
        self.nodes_visited_list.append(str(nodes_visited))
    def add_MST(self, MST):
        self.MST_list.append(MST)
    def write(self, name):
        text = '--------------------------------------\n-------------------------------------\n\n'
        for i in range(0, len(self.MST_list)):
            text += 'RONDA'+str(i)+':------------------------------------------\n'
            text += 'MST_'+str(i)+':' + str(self.MST_list[i]) + '\n-------------------------------------\n'
            text += 'Nodes_visited_'+str(i)+':' + str(self.nodes_visited_list[i])+ '\n-------------------------------------\n'
            text += '------Edges filtered----------'+str(i)+':\n' + str(self.filtered_edges_list[i])+ '\n-------------------------------------\n'
            text += '\n'+self.edges + '\n-------------------------------------\n-------------------------------------\n'
        with open( name, 'w') as f:
            f.write(text)