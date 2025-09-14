from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
ruta_subcarpeta = Path(Path(__file__).resolve().parent, 'SSGAs', 'data2')

#Calcula el coste medio por generacion y lo guarda
def crear_medias(nombre_archivo, h_limit, l_limit):

    ruta_archivo = Path(ruta_subcarpeta, nombre_archivo)

    df = pd.read_csv(ruta_archivo)

    # Comprobación de columnas esperadas
    if not {'Generacion', 'Fitness_max', 'Fitness_medio'}.issubset(set(df.columns)):
        raise ValueError(
            f"El archivo {ruta_archivo} no contiene las columnas requeridas 'Generacion','Fitness_max','Fitness_medio'.")

    # Agrupar por generación y calcular la media de ambas columnas
    df_grouped_mean = df.groupby('Generacion', as_index=False).agg({
        'Fitness_max': 'mean',
        'Fitness_medio': 'mean'
    })

    df_grouped_mean.to_csv(Path(ruta_subcarpeta, 'graficas', 'medias',nombre_archivo), index=False)
    crear_grafica(df_grouped_mean, nombre_archivo[:-4], h_limit, l_limit)

#Dibuja la grafica de convergencia y la guarda
def crear_grafica(df_mean, nombre, h_limit, l_limit):
    plt.figure(figsize=(8, 5))

    # Línea para Fitness medio
    plt.plot(df_mean["Generacion"], df_mean["Fitness_medio"], linestyle="-", label="Fitness medio")

    # Línea para Fitness máximo
    plt.plot(df_mean["Generacion"], df_mean["Fitness_max"], linestyle="--", label="Fitness máximo")

    # Añadir etiquetas y título
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Convergencia del Algoritmo Genético")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Guardar la gráfica
    plt.ylim(l_limit, h_limit)
    plt.savefig(Path(ruta_subcarpeta, 'graficas', nombre + '.png'))

def procesar_datos(h_limit, l_limit):
    archivos_csv = list(ruta_subcarpeta.glob("*.csv"))
    for archivo in archivos_csv:
        crear_medias(archivo.name, h_limit, l_limit)
procesar_datos(2200, 1200)