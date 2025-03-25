class FitnessDataLogger:
    def __init__(self, activated):
        self.edges = None
        self.activated = activated
        self.MST_list = []
        self.nodes_visited_list = []
        self.filtered_edges_list = []
        self.data = ''
    def add_edges(self, edges):
        if self.activated:
            self.edges = '------All edges----------\n'
            for edge in edges:
                self.edges += '-' + str(edge) + '-\n'
    def add_filtered_edges(self, edges):
        if self.activated:
            edges_filtered = ''
            for edge in edges:
                edges_filtered += '-' + str(edge) + '-\n'
            self.filtered_edges_list.append(edges_filtered)
    def add_nodes_visited(self, nodes_visited):
        if self.activated:
            nodes_visited_2 = dict(sorted(nodes_visited.items()))
            nodes_visited_2.copy()
            self.nodes_visited_list.append(str(nodes_visited_2))
    def add_MST(self, MST):
        if self.activated:
            self.MST_list.append(MST.copy())
    def write(self, name):
        if self.activated:
            text = '--------------------------------------\n-------------------------------------\n\n'
            for i in range(0, len(self.MST_list)):
                text += 'RONDA'+str(i)+':------------------------------------------\n'
                text += 'MST_'+str(i)+':' + str(self.MST_list[i]) + '\n-------------------------------------\n'
                text += 'Nodes_visited_'+str(i)+':' + str(self.nodes_visited_list[i])+ '\n-------------------------------------\n'
                text += '------Edges filtered----------'+str(i)+':\n' + str(self.filtered_edges_list[i])+ '\n-------------------------------------\n'
                text += '\n'+self.edges + '\n-------------------------------------\n-------------------------------------\n'
            with open( name, 'w') as f:
                f.write(text)