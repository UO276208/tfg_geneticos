import multiprocessing
from SSGAs import SSGA
from SSGAs.SSGA import fitness_fn_prim_hard_degree_limit
from util import lectorTSP


def crear_añadir_procesos(list_p, pop_number, fitness, graph, gen, i_, mut, name):
    proceso = multiprocessing.Process(target=SSGA.execute_genetic, args=(
        pop_number, fitness, graph, gen, i_, mut, name,))
    procesos.append(proceso)

    proceso.start()


if __name__ == "__main__":
    num_ejecuciones = 15
    procesos = []
    batch_size = 12  # Número de núcleos físicos
    prueba = lectorTSP.read_matrix("fri26.tsp")

    for i in range(num_ejecuciones):
        crear_añadir_procesos(procesos, 80, fitness_fn_prim_hard_degree_limit, prueba, 200, i, 0.05,
                              'prim_h_P80_G200_0.05-')
        crear_añadir_procesos(procesos, 80, fitness_fn_prim_hard_degree_limit, prueba, 200, i, 0.1,
                              'prim_h_P80_G200_0.1-')
        crear_añadir_procesos(procesos, 80, fitness_fn_prim_hard_degree_limit, prueba, 200, i, 0.15,
                              'prim_h_P80_G200_0.15-')
        crear_añadir_procesos(procesos, 80, fitness_fn_prim_hard_degree_limit, prueba, 200, i, 0.2,
                              'prim_h_P80_G200_0.2-')
        crear_añadir_procesos(procesos, 80, fitness_fn_prim_hard_degree_limit, prueba, 200, i, 0.25,
                              'prim_h_P80_G200_0.25-')

        # Esperar a que terminen los procesos del batch actual si ya hay 12 en ejecución
        if len(procesos) >= batch_size:
            for p in procesos:
                p.join()
            procesos = []  # Reiniciar la lista de procesos

    # Esperar a los procesos restantes
    for p in procesos:
        p.join()

    print("Todas las ejecuciones del algoritmo genético han terminado")
