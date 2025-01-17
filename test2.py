from functools import reduce

import numpy as np
decimals = 3
pop_number= 20
graph_size = 6

def get_number_distribution():
    gamma = 1.2
    n = np.random.normal(0, 1)
    return round((1 + gamma) ** n, decimals)


def init_population(map_maintain_solution, initial_chromosome, pop_number, graph_size):
    population = []

    for i in range(0,pop_number):
        population.append(fill_holes(map_maintain_solution, initial_chromosome))
    return population
def fill_holes(map_maintain_solution, chromosome):
    for i in range(0, len(chromosome)):
        if map_maintain_solution[i] == 0:
            chromosome[i] = get_number_distribution()
    return chromosome


print(get_number_distribution(0,0))
#Meter cromosomas del tamaño de las posiciones faltantes, en prim guardar el MST parcial para no tener que recalcularlo
# cada vez y solo valorar el coste de los nodos que se añaden nuevos (imagino que esto de igual ya que se ven reflejados en el coste total del arbol y la solucion a completar es la misma para todos)