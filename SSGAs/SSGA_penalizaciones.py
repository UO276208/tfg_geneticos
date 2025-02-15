from fitness import prim_sin_restriciones as pr, kruskal_sin_restricciones as kr
import numpy as np
from lectores_escritores import lectorTSP
from lectores_escritores import results_writer
import time

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

rw = results_writer.ResultsWriter()

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
    fitness = map(lambda chromosome: fitness_fn(chromosome, graph_matrix, ajusting=True), population)
    population_fitness = list(fitness)
    feasibles = []
    not_feasibles = []

    for pop in population_fitness:
        if pop[1] == 0:
            feasibles.append(pop)
        else:
            not_feasibles.append(pop)

    if len(not_feasibles) > 0:
        if len(feasibles) > 0:
            feasible = sorted(feasibles, key=lambda pop: pop[0], reverse=True)[0]
            not_feasible = sorted(not_feasibles, key=lambda pop: pop[0])[0]
            rw.add_fitness_f(feasible[0])
            rw.add_fitness_nf(not_feasible[0])
            penalty_coefficient = (feasible[0]-not_feasible[0])/(-(not_feasible[1])*not_feasible[0])
            if penalty_coefficient < 0:
                penalty_coefficient = 0
        else:#Esto no se si es buena idea, en teoria si no hay individuos que no violen restricciones
            # querría intentar que haya asi que incremento la penalizacion por cada violacion de restriccion en un 10%
            penalty_coefficient = penalty_coefficient * 1.1

#############################

def genetic_algorithm_stepwise(population, fitness_fn, graph_matrix, ngen=50, pmut=0.1):
    for generation in range(int(ngen)):
        #En intervalos de tamaño 5% de ngen se recalculan los coeficientes de penalizacion
        if generation % (ngen*0.05) == 0:
            ajust_penalty_coefficients(population,fitness_fn, graph_matrix)
        offspring = generate_offspring(population, fitness_fn, graph_matrix, pmut)
        population = replace_worst(population, offspring, fitness_fn, graph_matrix)
        best = max(population, key=lambda chromosome: fitness_fn(chromosome, graph_matrix))

        best_fitness = fitness_fn(best, graph_matrix)
        rw.add_fitness(best_fitness)

        print(str(best) + " Fitness:" + str(fitness_fn(best, graph_matrix)))
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
#print(genetic_algorithm_stepwise(init_population(50,len(prueba1)), fitness_fn_prim_penalty, prueba1,'pruuevba',ngen=90))
prueba = lectorTSP.read_matrix("fri26.tsp")
for i in range(0,10):
    print(str(i) + '------------------------------------------------------------------')
    inicio = time.time()
    print(genetic_algorithm_stepwise(init_population(80,len(prueba)), fitness_fn_kruskal_penalty, prueba,ngen=200))
    fin = time.time()
    rw.set_time(fin - inicio)
    rw.write('kruskal_nh_P80_G200-'+str(i))
#print(genetic_algorithm_stepwise(init_population(80,len(prueba)), fitness_fn_prim_penalty, prueba,'prim_P80_G200',ngen=200))
#print(genetic_algorithm_stepwise(init_population(80,len(prueba)), fitness_fn_kruskal_penalty, prueba,'kruskal_P80_G200',ngen=200))
