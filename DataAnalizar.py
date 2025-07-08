from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
ruta_subcarpeta = Path(Path(__file__).resolve().parent, 'SSGAs', 'data2')

#Calcula el coste medio por generacion y lo guarda
def crear_medias(nombre_archivo, h_limit, l_limit):

    ruta_archivo = Path(ruta_subcarpeta, nombre_archivo)

    df = pd.read_csv(ruta_archivo)

    df_grouped_mean = df.groupby('Generacion', as_index=False)["Fitness"].mean()
    df_grouped_mean.to_csv(Path(ruta_subcarpeta, 'graficas', 'medias',nombre_archivo), index=False)
    crear_grafica(df_grouped_mean, nombre_archivo[:-4], h_limit, l_limit)

#Dibuja la grafica de convergencia y la guarda
def crear_grafica(df_mean, nombre, h_limit, l_limit):
    plt.figure(figsize=(8, 5))
    plt.plot(df_mean["Generacion"], df_mean["Fitness"], marker="o", linestyle="-", color="steelblue", label="Fitness medio")

    # Añadir etiquetas y título
    plt.xlabel("Generación")
    plt.ylabel("Fitness medio")
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
procesar_datos(1600, 800)