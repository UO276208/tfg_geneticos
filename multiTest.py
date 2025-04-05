import multiprocessing
from SSGAs import SSGA
from SSGAs.SSGA import fitness_fn_prim_hard_degree_limit
from util import lectorTSP

if __name__ == "__main__":
    num_ejecuciones = 15
    procesos = []
    batch_size = 12  # Número de núcleos físicos
    prueba = lectorTSP.read_matrix("fri26.tsp")

    for i in range(num_ejecuciones):
        proceso = multiprocessing.Process(target=SSGA.execute_genetic, args=(80,fitness_fn_prim_hard_degree_limit,prueba,200,i,0.1,))
        procesos.append(proceso)

        proceso.start()

        # Esperar a que terminen los procesos del batch actual si ya hay 12 en ejecución
        if len(procesos) >= batch_size:
            for p in procesos:
                p.join()
            procesos = []  # Reiniciar la lista de procesos

    # Esperar a los procesos restantes
    for p in procesos:
        p.join()

    print("Todas las ejecuciones del algoritmo genético han terminado")