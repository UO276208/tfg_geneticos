from fitness import prim_sin_restriciones as pr, kruskal_sin_restricciones as kr
import numpy as np
from util import results_writer, lectorTSP
import time
from util import DataLogger

prueba1 = [[0, 4, 3, 9],
           [4, 0, 8, 10],
           [3, 8, 1, 1],
           [9, 10, 1, 0]]
prueba2 = [0,
           64, 0,
           378, 318, 0,
           519, 455, 170, 0,
           434, 375, 265, 223, 0,
           200, 164, 344, 428, 273, 0]
k = 4
number_of_sons = 2
decimals = 3
tournament_size = 0.2
penalty_coefficient = 0.1

#rw = results_writer.ResultsWriter()

#Variable global que guarde el coheficiente de penalizacion
#Cada 10 (por ejemplo) generaciones sacar un vector con los fitness de la poblacion de la generación actual, meterlo en un diccionario
#de manera que cada individuo sea la clave y su fitness el valor (adjuntar el numero de violaciones de restriccion
# que lo devuelve prim a la funcion fitness) dividirlo en dos diccionarios, uno con los individuos que no violen ninguna restriccion y otro
#con el resto para poder sacar el peor de los primeros (los correctos) y el mejor de los malos (lo que violan alguna restriccion)
#############################
def fitness_fn_prim_penalty(sample, graph_matrix_ft, ajusting=False):
    graph = pr.Graph_prim(graph_matrix_ft, sample, k, input_data_type='gm')
    mst, n_violations = graph.prim()
    real_cost = 0
    for edge in mst:
        real_cost += graph_matrix_ft[edge[1]][edge[2]]
    if ajusting:
        return real_cost, n_violations
    else:
        return real_cost + (n_violations * (real_cost*penalty_coefficient)) #Temporal, no se si es la mejor manera de aplicarlo


def fitness_fn_kruskal_penalty(sample, graph_matrix_ft, ajusting=False):
    graph = kr.Graph_kruskal(graph_matrix_ft, sample, k, input_data_type='gm')
    mst, n_violations = graph.kruskal()
    real_cost = 0
    for edge in mst:
        real_cost += graph_matrix_ft[edge[1]][edge[2]]

    if ajusting:
        return real_cost, n_violations
    else:
        return real_cost + (n_violations * (real_cost*penalty_coefficient)) #Temporal, no se si es la mejor manera de aplicarlo

#############################
def ajust_penalty_coefficients(population, fitness_fn, graph_matrix):
    global penalty_coefficient

    feasibles = []
    not_feasibles = []

    for pop in population:
        pop_ajust = fitness_fn(pop[0], graph_matrix, True)
        if pop_ajust[1] == 0:
            feasibles.append(pop)
        else:
            not_feasibles.append(pop)

    if len(not_feasibles) > 0:
        if len(feasibles) > 0:
            feasible = sorted(feasibles, key=lambda chromosome: chromosome[1], reverse=True)[0][1]
            not_feasible = sorted(not_feasibles, key=lambda chromosome: chromosome[1])[0][1]
            penalty_coefficient = (feasible-not_feasible)/(-(not_feasible)*not_feasible)
            if penalty_coefficient < 0:
                penalty_coefficient = 0
        else:#Esto no se si es buena idea, en teoria si no hay individuos que no violen restricciones
            # querría intentar que haya asi que incremento la penalizacion por cada violacion de restriccion en un 10%
            penalty_coefficient = penalty_coefficient * 1.1

#############################

def genetic_algorithm_stepwise(rw ,population, fitness_fn, graph_matrix, ngen=50, pmut=0.1):
    for generation in range(int(ngen)):
        #En intervalos de tamaño 5% de ngen se recalculan los coeficientes de penalizacion
        if generation % (ngen*0.05) == 0:
            ajust_penalty_coefficients(population,fitness_fn, graph_matrix)
        offspring = generate_offspring(population, fitness_fn, pmut, graph_matrix)
        population = replace_worst(population, offspring)

        best = max(population, key=lambda chromosome: chromosome[1])

        best_fitness = best[1]
        rw.add_fitness(best_fitness)

        print(str(best[0]) + " Fitness:" + str(best_fitness))
    return max(population, key=lambda chromosome: chromosome[1])


def replace_worst(population, offspring):
    ordered_population = sorted(population, key=lambda chromosome: chromosome[1], reverse=True)
    for i in range(0, len(offspring)):
        if offspring[i][1] < ordered_population[i][1]:
            ordered_population[i] = offspring[i]
    return ordered_population


def generate_offspring(population, fitness_fn, pmut, graph_matrix):
    offspring = []
    x, y = tournament(population)
    i = 0
    while len(offspring) < number_of_sons and i <= 10:
        son = uniform_crossover(x, y, fitness_fn, graph_matrix)
        i += 1
        if not (son in offspring):
            i -= 1
            offspring.append(mutate(son, pmut, fitness_fn, graph_matrix))

    return offspring


def mutate(x, pmut, fitness_fn, graph_matrix):
    if np.random.rand() >= pmut:
        return x
    i = np.random.randint(0, (len(x[0]) - 1))
    x[0][i] = get_number_distribution(0,0)
    x[1] = fitness_fn(x[0], graph_matrix)
    return x

def uniform_crossover(x, y, fitness_fn, graph_matrix):
    probability_vector = np.random.randint(2, size=len(x[0]))
    child = []
    for i in range(0, len(probability_vector)):
        if probability_vector[i] == 0:
            child.append(x[0][i])
        elif probability_vector[i] == 1:
            child.append(y[0][i])
    return [child, fitness_fn(child, graph_matrix)]


def get_number_distribution(i, j):
    gamma = 1.2
    n = np.random.normal(0, 1)
    return round((1 + gamma) ** n, decimals)


def init_population(pop_number, graph, fitness):
    graph_size = len(graph)
    array = np.fromfunction(np.vectorize(get_number_distribution), (pop_number, graph_size + 1), dtype=float)
    population = []

    control_gene = [0] * (graph_size + 1)
    control_gene = list(map(lambda i: i + 1, control_gene))
    control_gene[len(control_gene) - 1] = 0
    population.append((control_gene, fitness(control_gene,graph)))

    for gene in array:
        gene_list = gene.tolist()
        gene_list[len(gene) - 1] = np.random.randint(0, graph_size)
        population.append([gene_list, fitness(gene_list, graph)])
    return population


def tournament(population):
    parents = [get_winers(population), get_winers(population)]

    return parents


def get_winers(population):
    window_size = round(len(population) * tournament_size)
    start_point_window = np.random.randint(0, len(population) - window_size)

    competitors = get_competitors(population, start_point_window, window_size, 6)
    ordered_competitors = sorted(competitors, key=lambda chromosome: chromosome[1])

    return ordered_competitors[0]


def get_competitors(population, start_point_window, window_size, number_of_competitors):
    competitors = []
    i = 0
    attempts = 0
    while i < number_of_competitors and attempts < 10:
        competitor = population[np.random.randint(start_point_window, window_size + start_point_window)]
        attempts += 1
        if not (competitor in competitors):
            i += 1
            attempts -= 1
            competitors.append(competitor)

    return competitors

def execute_genetic(pop_number, fitness, graph, gen, i, mut, name):
    rw = DataLogger.DataLogger(gen)
    inicio = time.time()
    genetic_algorithm_stepwise(rw, init_population(pop_number, graph, fitness), fitness, graph, ngen=gen, pmut=mut)
    fin = time.time()
    rw.set_time(fin - inicio)
    rw.write(name+str(i),name[:-1])

prueba = lectorTSP.read_matrix("fri26.tsp")
grafo = [[0,4,10,3,2],
         [4, 0, 1, 5, 1],
         [10,1, 0, 2, 4],
         [3, 5, 2, 0 ,6],
         [2, 1, 4, 6, 0]]
matriz_adyacencia = [
    [0, 3, 7, 2, 5, 9, 1, 4, 8, 6],
    [3, 0, 6, 4, 8, 2, 7, 5, 9, 1],
    [7, 6, 0, 5, 3, 8, 2, 9, 4, 10],
    [2, 4, 5, 0, 7, 6, 3, 8, 1, 9],
    [5, 8, 3, 7, 0, 4, 6, 2, 10, 1],
    [9, 2, 8, 6, 4, 0, 5, 3, 7, 10],
    [1, 7, 2, 3, 6, 5, 0, 10, 9, 4],
    [4, 5, 9, 8, 2, 3, 10, 0, 6, 7],
    [8, 9, 4, 1, 10, 7, 9, 6, 0, 2],
    [6, 1, 10, 9, 1, 10, 4, 7, 2, 0]]
matriz_adyacencia2 = [
    [0, 3, 7, 2, 5, 9, 1, 0, 8, 6],
    [3, 0, 6, 4, 8, 2, 7, 5, 9, 1],
    [7, 6, 0, 5, 3, 8, 2, 9, 4, 0],
    [2, 0, 5, 0, 7, 6, 3, 8, 1, 9],
    [5, 8, 3, 7, 0, 4, 6, 2, 10, 1],
    [9, 2, 8, 6, 4, 0, 5, 3, 7, 10],
    [1, 7, 2, 3, 6, 5, 0, 10, 9, 4],
    [4, 0, 9, 8, 2, 3, 10, 0, 6, 7],
    [8, 9, 4, 1, 10, 0, 9, 6, 0, 2],
    [6, 1, 10, 9, 1, 10, 4, 7, 2, 0]]
execute_genetic(100,fitness_fn_prim_penalty,grafo,5,1,0.05,'prim_h_P80_G200_0.05-')

#print(genetic_algorithm_stepwise(init_population(80,len(prueba)), fitness_fn_prim_hard_degree_limit, prueba, 'kruskal_h_P80_G200', ngen=200))