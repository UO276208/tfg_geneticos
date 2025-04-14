import multiprocessing
import os
import pandas as pd

from SSGAs import SSGA
from SSGAs.SSGA import fitness_fn_prim_hard_degree_limit
from util import lectorTSP

def lanzar_test(num_ejecuciones, batch_size, pop_number, fitness, graph, gen, mut, name):
    nombre = name + f'P{pop_number}_G{gen}_{mut}-'
    procesos = []
    for i in range(num_ejecuciones):
        proceso = multiprocessing.Process(target=SSGA.execute_genetic, args=(
            pop_number, fitness, graph, gen, i, mut, nombre,))
        procesos.append(proceso)

        proceso.start()

        # Esperar a que terminen los procesos del batch actual si ya hay 12 en ejecución
        if len(procesos) >= batch_size:
            for p in procesos:
                p.join()

    # Esperar a los procesos restantes
    for p in procesos:
        p.join()

def unir_csvs(output_dir, nombre_final='resultados_unificados.csv'):
    '''
    TODO

    :param output_dir:
    :param nombre_final:
    :return:
    '''
    archivos = [f for f in os.listdir(output_dir) if f.startswith("ejecucion_") and f.endswith(".csv")]
    df_total = pd.concat([pd.read_csv(os.path.join(output_dir, f)).assign(run_id=i)
                         for i, f in enumerate(sorted(archivos))], ignore_index=True)
    df_total.to_csv(os.path.join(output_dir, nombre_final), index=False)
if __name__ == "__main__":
    prueba = lectorTSP.read_matrix("fri26.tsp")

    lanzar_test(4, 6,30, fitness_fn_prim_hard_degree_limit, prueba, 50, 0.05,
                              'prim_h_')
    lanzar_test(4, 6, 30, fitness_fn_prim_hard_degree_limit, prueba, 50, 0.15,
                'prim_h_')

    print("Todas las ejecuciones del algoritmo genético han terminado")
