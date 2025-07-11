import multiprocessing

from pathlib import Path

import pandas as pd

from SSGAs import SSGA
from SSGAs import SSGA_penalizaciones
from SSGAs.SSGA_penalizaciones import fitness_fn_prim_penalty
from util import lectorTSP

def lanzar_test(SSGA_version, num_ejecuciones, batch_size, pop_number, fitness, graph, gen, mut, name):
    nombre = name + f'P{pop_number}_G{gen}_{mut}-'
    procesos = []
    for i in range(num_ejecuciones):
        proceso = multiprocessing.Process(target=SSGA_version.execute_genetic, args=(
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
    prueba = lectorTSP.read_matrix("fri26.tsp")

    #lanzar_test(SSGA,20, 12,100, fitness_fn_prim_hard_degree_limit, prueba, 200, 0.03,'prim_imp_h_')
    lanzar_test(SSGA_penalizaciones, 20, 12, 100, fitness_fn_prim_penalty, prueba, 200, 0.03, 'prim_h_pena')

    print("Todas las ejecuciones del algoritmo genético han terminado")
