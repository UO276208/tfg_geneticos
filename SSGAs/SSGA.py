from fitness import kruskal as kr, prim
import numpy as np
from lectores_escritores import lectorTSP
from lectores_escritores import results_writer

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
k = 2
number_of_sons = 2
decimals = 3
tournament_size = 0.2
rw = results_writer.ResultsWriter()


def fitness_fn_prim_hard_degree_limit(sample, graph_matrix_ft):
    graph = prim.Graph_prim(graph_matrix_ft, sample, k, input_data_type='gm')
    mst = graph.prim()
    real_cost = 0
    for edge in mst:
        real_cost += graph_matrix_ft[edge[1]][edge[2]]
    return real_cost


def fitness_fn_kruskal_hard_degree_limit(sample, graph_matrix_ft):
    graph = kr.Graph_kruskal(graph_matrix_ft, sample, k, input_data_type='gm')
    mst = graph.kruskal()
    real_cost = 0
    for edge in mst:
        real_cost += graph_matrix_ft[edge[1]][edge[2]]
    return real_cost


#############################
#############################

def genetic_algorithm_stepwise(population, fitness_fn, graph_matrix, file_name, ngen=50, pmut=0.1):
    for generation in range(int(ngen)):
        offspring = generate_offspring(population, fitness_fn, graph_matrix, pmut)
        population = replace_worst(population, offspring, fitness_fn, graph_matrix)
        best = max(population, key=lambda chromosome: fitness_fn(chromosome, graph_matrix))
        best_fitness = fitness_fn(best, graph_matrix)
        rw.add_fitness(best_fitness)
        print('Gen ' + str(generation) + ': ' + str(best) + " Fitness:" + str(best_fitness))
    rw.write(file_name)
    return max(population, key=lambda chromosome: fitness_fn(chromosome, graph_matrix))


def replace_worst(population, offspring, fitness_fn, graph_matrix):
    ordered_population = sorted(population, key=lambda chromosome: fitness_fn(chromosome, graph_matrix), reverse=True)
    for i in range(0, len(offspring)):
        if fitness_fn(offspring[i], graph_matrix) < fitness_fn(ordered_population[i], graph_matrix):
            ordered_population[i] = offspring[i]
    return ordered_population


def generate_offspring(population, fitness_fn, graph_matrix, pmut):
    offspring = []
    x, y = tournament(population, fitness_fn, graph_matrix)
    i = 0
    while len(offspring) < number_of_sons and i <= 10:
        son = uniform_crossover(x, y)
        i += 1
        if not (son in offspring):
            i -= 1
            offspring.append(mutate(son, pmut))

    return offspring


def mutate(x, pmut):
    if np.random.rand() >= pmut:
        return x
    i = np.random.randint(0, (len(x) - 2))
    x[i] = 1
    return x

def uniform_crossover(x, y):
    probability_vector = np.random.randint(2, size=len(x))
    child = []
    for i in range(0, len(probability_vector)):
        if probability_vector[i] == 0:
            child.append(x[i])
        elif probability_vector[i] == 1:
            child.append(y[i])
    return child


def get_number_distribution(i, j):
    gamma = 1.2
    n = np.random.normal(0, 1)
    return round((1 + gamma) ** n, decimals)


def init_population(pop_number, graph_size):
    array = np.fromfunction(np.vectorize(get_number_distribution), (pop_number, graph_size + 1), dtype=float)
    population = []

    control_gene = [0] * (graph_size + 1)
    control_gene = list(map(lambda i: i + 1, control_gene))
    control_gene[len(control_gene) - 1] = 0
    population.append(control_gene)

    for gene in array:
        gene_list = gene.tolist()
        gene_list[len(gene) - 1] = np.random.randint(0, graph_size)
        population.append(gene_list)
    return population


def tournament(population, fitness_fn, graph_matrix):
    parents = []

    parents.append(get_winers(population, fitness_fn, graph_matrix))
    parents.append(get_winers(population, fitness_fn, graph_matrix))

    return parents


def get_winers(population, fitness_fn, graph_matrix):
    window_size = round(len(population) * tournament_size)
    start_point_window = np.random.randint(0, len(population) - window_size)

    competitors = get_competitors(population, start_point_window, window_size, 6)
    ordered_competitors = sorted(competitors, key=lambda chromosome: fitness_fn(chromosome, graph_matrix))

    return ordered_competitors[0]


def get_competitors(population, start_point_window, window_size, number_of_competitors):
    competitors = []
    i = 0
    j = 0
    while i < number_of_competitors and j < 10:
        competitor = population[np.random.randint(start_point_window, window_size + start_point_window)]
        j += 1
        if not (competitor in competitors):
            i += 1
            j -= 1
            competitors.append(competitor)

    return competitors




# print(genetic_algorithm_stepwise( [chromosome1, chromosome2, chromosome3, chromosome4, chromosome5, chromosome6, chromosome0], fitness_fn))
# print(get_parents([chromosome1,chromosome2,chromosome3],fitness_fn, graph_matrix))
#print(genetic_algorithm_stepwise(init_population(50,len(prueba1)), fitness_fn_prim_hard_degree_limit, prueba1,ngen=90))
#print(genetic_algorithm_stepwise(init_population(50, len(prueba1)), fitness_fn_kruskal_hard_degree_limit, prueba1, ngen=90))
prueba = lectorTSP.read_matrix("fri26.tsp")
print(genetic_algorithm_stepwise(init_population(80,len(prueba)), fitness_fn_prim_hard_degree_limit, prueba, 'prim_h_P80_G200',ngen=200))
print(genetic_algorithm_stepwise(init_population(80,len(prueba)), fitness_fn_kruskal_hard_degree_limit, prueba, 'kruskal_h_P80_G200', ngen=200))
