import multiprocessing
import time

from pathlib import Path

import pandas as pd

from SSGAs import SSGA
from SSGAs import SSGA_penalizaciones
from SSGAs.SSGA import fitness_fn_prim_hard_degree_limit, fitness_fn_kruskal_hard_degree_limit
from SSGAs.SSGA_penalizaciones import fitness_fn_prim_penalty
from util import lectorTSP

def lanzar_test(SSGA_version, num_ejecuciones, batch_size, pop_number, fitness, graph, gen, mut, gamma_param, name):
    nombre = name + f'P{pop_number}_G{gen}_{mut}-gamma_{gamma_param}-'
    procesos = []
    for i in range(num_ejecuciones):
        proceso = multiprocessing.Process(target=SSGA_version.execute_genetic, args=(
            pop_number, fitness, graph, gen, i, mut, nombre, gamma_param, ))
        procesos.append(proceso)

        proceso.start()

        # MODIFIED: esperar sólo mientras haya >= batch_size procesos VIVOS;
        #            eliminar de la lista los que ya terminaron (y hacer join).
        while len(procesos) >= batch_size:
            # Recorremos una copia para poder eliminar mientras iteramos
            for p in procesos[:]:
                if not p.is_alive():
                    p.join()           # ADDED: recoger recursos del proceso terminado
                    procesos.remove(p) # ADDED: quitar de la lista de procesos en ejecución
            if len(procesos) >= batch_size:
                time.sleep(0.1)      # ADDED: evitar busy-waiting

    # Esperar a los procesos restantes
    for p in procesos:
        p.join()
    unir_csvs(Path(Path(__file__).resolve().parent, 'SSGAs', 'data2', nombre[:-1]), nombre[:-1]+'.csv')

def unir_csvs(output_dir, nombre_final):
    # Listar todos los archivos CSV en la subcarpeta
    archivos_csv = list(output_dir.glob("*.csv"))

    # Crear una lista para almacenar los DataFrames de cada archivo
    lista_df = []
    i = 0
    for archivo in archivos_csv:
        i+=1
        df = pd.read_csv(archivo)
        # Agrega una columna que identifique de qué archivo proviene el DataFrame
        df['Fuente'] = i
        lista_df.append(df)
    # Unir todos los DataFrames en uno solo
    df_consolidado = pd.concat(lista_df, ignore_index=True)

    # Definir la ruta completa para el archivo resultante
    ruta_resultante = Path(output_dir.parent,nombre_final)

    # Guardar el DataFrame consolidado en el archivo CSV
    df_consolidado.to_csv(ruta_resultante, index=False)
    #Eliminar los archivos individuales
    for archivo in archivos_csv:
        archivo.unlink()
    #Eliminar el directorio donde estaban esos archivos
    Path(output_dir).rmdir()

if __name__ == "__main__":
    fri26 = lectorTSP.read_matrix("fri26.tsp")
    lin318 = lectorTSP.read_matrix("lin318.tsp")
    si535 = lectorTSP.read_matrix("si535.tsp")
    bayg29 = lectorTSP.read_matrix("bayg29.tsp")

    #limit gen
    lanzar_test(SSGA, 30, 12, 1500, fitness_fn_kruskal_hard_degree_limit, bayg29, 100000, 0.03, 0.3, 'kruskal_imp_h_bayg29_limitgentest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 100000, 0.03, 0.3, 'kruskal_imp_h_bayg29_limitgentest')
    lanzar_test(SSGA, 30, 12, 3000, fitness_fn_kruskal_hard_degree_limit, bayg29, 100000, 0.03, 0.3, 'kruskal_imp_h_bayg29_limitgentest')
    lanzar_test(SSGA, 30, 12, 5000, fitness_fn_kruskal_hard_degree_limit, bayg29, 100000, 0.03, 0.3, 'kruskal_imp_h_bayg29_limitgentest')

    #pop
    lanzar_test(SSGA, 30, 12, 200, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.3,'kruskal_imp_h_bayg29_poptest')
    lanzar_test(SSGA, 30, 12, 500, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.3,'kruskal_imp_h_bayg29_poptest')
    lanzar_test(SSGA, 30, 12, 1000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.3,'kruskal_imp_h_bayg29_poptest')
    lanzar_test(SSGA, 30, 12, 1500, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.3,'kruskal_imp_h_bayg29_poptest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.3,'kruskal_imp_h_bayg29_poptest')
    lanzar_test(SSGA, 30, 12, 3000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.3,'kruskal_imp_h_bayg29_poptest')
    lanzar_test(SSGA, 30, 12, 5000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.3,'kruskal_imp_h_bayg29_poptest')

    #mut
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.01, 0.3, 'kruskal_imp_h_bayg29_muttest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.02, 0.3, 'kruskal_imp_h_bayg29_muttest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.3, 'kruskal_imp_h_bayg29_muttest_gamma')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.04, 0.3, 'kruskal_imp_h_bayg29_muttest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.05, 0.3, 'kruskal_imp_h_bayg29_muttest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.06, 0.3, 'kruskal_imp_h_bayg29_muttest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.07, 0.3, 'kruskal_imp_h_bayg29_muttest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.08, 0.3, 'kruskal_imp_h_bayg29_muttest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.09, 0.3, 'kruskal_imp_h_bayg29_muttest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.1, 0.3, 'kruskal_imp_h_bayg29_muttest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.15, 0.3, 'kruskal_imp_h_bayg29_muttest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.2, 0.3, 'kruskal_imp_h_bayg29_muttest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.25, 0.3, 'kruskal_imp_h_bayg29_muttest')

    #gamma
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.0, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.1, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.2, 'kruskal_imp_h_bayg29_gammatest')
    #lanzar_test(SSGA, 30, 12, 2000, fitness_fn_prim_hard_degree_limit, bayg29, 20000, 0.03, 0.3, 'kruskal_imp_h_bayg29') Este ya está arriba en mut
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.4, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.5, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.6, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.7, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.8, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 0.9, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 1.0, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 1.1, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 1.2, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 1.3, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 1.4, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 1.5, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 1.6, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 1.7, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 1.8, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 1.9, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 2.0, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 2.1, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 2.2, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 2.3, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 2.4, 'kruskal_imp_h_bayg29_gammatest')
    lanzar_test(SSGA, 30, 12, 2000, fitness_fn_kruskal_hard_degree_limit, bayg29, 20000, 0.03, 2.5, 'kruskal_imp_h_bayg29_gammatest')





    #lanzar_test(SSGA, 30, 12, 2000, fitness_fn_prim_hard_degree_limit, bayg29, 20000, 0.03, 0.3, 'prim_imp_h_bayg29')
    #lanzar_test(SSGA, 30, 12, 2000, fitness_fn_prim_hard_degree_limit, bayg29, 20000, 0.03, 0.3, 'prim_imp_h_bayg29')



    #lanzar_test(SSGA_penalizaciones, 20, 12, 100, fitness_fn_prim_penalty, prueba, 200, 0.03, 'prim_h_pena')

    print("Todas las ejecuciones del algoritmo genético han terminado")
