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
#penalty_coefficient = 0.1


#############################
def fitness_fn_prim_penalty(sample, graph_matrix_ft):
    graph = pr.Graph_prim(graph_matrix_ft, sample, k, input_data_type='gm')
    mst, n_violations = graph.prim()
    real_cost = 0
    for edge in mst:
        real_cost += graph_matrix_ft[edge[1]][edge[2]]
    return real_cost, n_violations
    #if ajusting:
        #return real_cost, n_violations
    #else:
        #return real_cost + (n_violations * (real_cost*penalty_coefficient)) #Temporal, no se si es la mejor manera de aplicarlo


def fitness_fn_kruskal_penalty(sample, graph_matrix_ft):
    graph = kr.Graph_kruskal(graph_matrix_ft, sample, k, input_data_type='gm')
    mst, n_violations = graph.kruskal()
    real_cost = 0
    for edge in mst:
        real_cost += graph_matrix_ft[edge[1]][edge[2]]
    return real_cost, n_violations
    #if ajusting:
        #return real_cost, n_violations
    #else:
    # return real_cost + (n_violations * (real_cost*penalty_coefficient)) #Temporal, no se si es la mejor manera de aplicarlo

#############################
def ajust_penalty_coefficients(population_set, penalty_coefficient):

    feasibles = []
    not_feasibles = []

    for pop in population_set:
        if pop[1][1] == 0:
            feasibles.append(pop)
        else:
            not_feasibles.append(pop)

    if len(not_feasibles) > 0:
        if len(feasibles) > 0:
            feasible = sorted(feasibles, key=lambda chromosome: chromosome[1][0], reverse=True)[0][1][0]
            not_feasible = sorted(not_feasibles, key=lambda chromosome: chromosome[1][0])[0][1][0]
            penalty_coefficient = (feasible-not_feasible) / ( -not_feasible * not_feasible)
            if penalty_coefficient < 0:
                penalty_coefficient = 0
        #else:#Esto no se si es buena idea, en teoria si no hay individuos que no violen restricciones
            # querría intentar que haya asi que incremento la penalizacion por cada violacion de restriccion en un 10%
            #penalty_coefficient = penalty_coefficient * 1.1
    return penalty_coefficient

#############################

def genetic_algorithm_stepwise(rw ,population, fitness_fn, graph_matrix,
                               population_set, distrib_param, penalty_coefficient_param, ngen=50, pmut=0.1):
    penalty_coefficient = penalty_coefficient_param
    for generation in range(int(ngen)):
        #En intervalos de tamaño 5% de ngen se recalculan los coeficientes de penalizacion
        if generation % (ngen*0.05) == 0:
            penalty_coefficient = ajust_penalty_coefficients(population_set, penalty_coefficient)

        offspring = generate_offspring(population, pmut, distrib_param)
        population = replace_worst(population, offspring, population_set, fitness_fn, graph_matrix, penalty_coefficient)

        best = max(population, key=lambda chromosome: chromosome[1])

        fitness_array = np.array([chromo[1] for chromo in population])
        fitness_avg = np.mean(fitness_array)

        rw.add_fitness(best[1], fitness_avg)
        print('Gen ' + str(generation) + ': ' + str(best) + " Fitness:" + str(best[1]))
    return min(population, key=lambda chromosome: chromosome[1])


def replace_worst(population, offspring, population_set, fitness_fn, graph_matrix, penalty_coefficient):
    ordered_population = sorted(population, key=lambda chromosome: chromosome[1], reverse=True)
    for i in range(0, len(offspring)):
        #Aplicar la penalizacion
        chromosome_fitness, chromosome_n_violation = fitness_fn(offspring[i],graph_matrix)
        chromosome_fitness = (chromosome_fitness +
                                      (chromosome_n_violation * (chromosome_fitness*penalty_coefficient)))
        chromosome_tuppled = (tuple(offspring[i]), tuple((chromosome_fitness, chromosome_n_violation)))
        chromosome = [offspring[i], chromosome_fitness]

        if chromosome_fitness < ordered_population[i][1] and not chromosome_tuppled in population_set:
            population_set.add(chromosome_tuppled)
            ordered_population[i] = chromosome
    return ordered_population


def generate_offspring(population, pmut, distrib_param):
    offspring = []
    x, y = tournament(population)
    i = 0
    while len(offspring) < number_of_sons and i <= 10:
        son = uniform_crossover(x, y)
        i += 1
        if not (son in offspring):
            i -= 1
            offspring.append(mutate(son, pmut, distrib_param))

    return offspring


def mutate(x, pmut, distrib_param):
    if np.random.rand() >= pmut:
        return x
    i = np.random.randint(0, (len(x) - 1))
    x[i] = get_number_distribution(0,0, distrib_param)

    return x


def uniform_crossover(x, y):
    probability_vector = np.random.randint(2, size=len(x[0]))
    child = []
    for i in range(0, len(probability_vector)):
        if probability_vector[i] == 0:
            child.append(x[0][i])
        elif probability_vector[i] == 1:
            child.append(y[0][i])
    return child


def get_number_distribution(i, j, gamma):
    n = np.random.normal(0, 1)
    return round((1 + gamma) ** n, decimals)

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

def init_population(pop_number, graph, fitness, distrib_param):
    graph_size = len(graph)
    array = np.fromfunction(np.vectorize(lambda i, j: get_number_distribution(i, j, distrib_param)),
                            (pop_number, graph_size + 1), dtype=float)

    population = []
    population_set = set()

    control_gene = [0] * (graph_size + 1)
    control_gene = list(map(lambda i: i + 1, control_gene))
    control_gene[len(control_gene) - 1] = np.random.randint(0, graph_size)
    control_gene_fitness_data = fitness(control_gene,graph)

    population_set.add((tuple(control_gene), tuple(control_gene_fitness_data)))
    population.append((control_gene, control_gene_fitness_data[0]))

    for gene in array:
        gene_list = gene.tolist()
        gene_list[len(gene) - 1] = np.random.randint(0, graph_size)
        gene_fitness_data = fitness(gene_list, graph)

        population_set.add((tuple(gene_list), tuple(gene_fitness_data)))
        population.append([gene_list, gene_fitness_data[0]])

    return population, population_set

def execute_genetic(pop_number, fitness, graph, gen, i, mut, name, distrib_param):
    rw = DataLogger.DataLogger(gen)
    inicio = time.time()
    population, population_set = init_population(pop_number, graph, fitness, distrib_param)

    genetic_algorithm_stepwise(rw, population, fitness, graph, population_set,distrib_param,0.1, ngen=gen, pmut=mut)
    fin = time.time()
    rw.set_time(fin - inicio)
    rw.write(name+str(i),name[:-1])

prueba = lectorTSP.read_matrix("fri26.tsp")
bayg29 = lectorTSP.read_matrix("bayg29.tsp")
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

execute_genetic(2000,fitness_fn_prim_penalty,bayg29,20000,1,0.03,'prueba-AA', 0.3)
