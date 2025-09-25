from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt


#Calcula el coste medio por generacion y lo guarda
def crear_medias(ruta_subcarpeta, nombre_archivo):

    ruta_archivo = Path(ruta_subcarpeta, nombre_archivo)
    ruta_salida = Path(ruta_subcarpeta, 'medias')
    ruta_salida.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ruta_archivo)

    # Comprobación de columnas esperadas
    if not {'Generacion', 'Fitness_max', 'Fitness_medio', 'Tiempo'}.issubset(df.columns):
        raise ValueError(
            f"El archivo {ruta_archivo} no contiene las columnas requeridas "
            "'Generacion','Fitness_max','Fitness_medio','Tiempo'."
        )

    # Agrupar por generación y calcular la media de todas las columnas relevantes
    df_grouped_mean = df.groupby('Generacion', as_index=False).agg({
        'Fitness_max': 'mean',
        'Fitness_medio': 'mean',
        'Tiempo': 'mean'
    })

    #Guardar datos de las medias
    df_grouped_mean.to_csv(Path(ruta_salida, nombre_archivo), index=False)

    # Fila primera y última
    primera_gen = df_grouped_mean.iloc[0]
    ultima_gen = df_grouped_mean.iloc[-1]

    return pd.DataFrame([{
        "archivo": nombre_archivo,
        "fitness_ini": primera_gen["Fitness_medio"],
        "fitness_max": ultima_gen["Fitness_max"],
        "fitness_medio": ultima_gen["Fitness_medio"],
        "tiempo": ultima_gen["Tiempo"]
    }])

def procesar_diretorio(ruta):
    archivos = list(ruta.glob("*.csv"))
    resumen = []

    for archivo in archivos:
        df_resumen = crear_medias(ruta, archivo.name)
        resumen.append(df_resumen )
    if resumen:
        df_resultados = pd.concat(resumen, ignore_index=True)
        ruta_salida = Path(ruta, 'medias')
        ruta_salida.mkdir(parents=True, exist_ok=True)
        df_resultados.to_csv(ruta_salida / "resumen.csv", index=False)
        return df_resultados
    return pd.DataFrame(columns=["archivo", "fitness_max", "fitness_medio", "tiempo"])

def procesar_datos():
    ruta_subcarpeta = Path(Path(__file__).resolve().parent, 'SSGAs', 'data_copia')
    sub_carpeta_resumen = []

    for sub in ruta_subcarpeta.iterdir():
        if sub.is_dir():
            print(f"Analizando datos de {sub.name}:")
            for subsub in sub.iterdir():
                if subsub.is_dir():
                    print("   Test :", subsub.name, )
                    resumen = procesar_diretorio(Path(sub, subsub))
                    resumen["subcarpeta"] = sub.name
                    resumen["subsubcarpeta"] = subsub.name
                    sub_carpeta_resumen.append(resumen)

    if sub_carpeta_resumen:
        # Concatenar todos los resúmenes en un único DataFrame
        df_total = pd.concat(sub_carpeta_resumen, ignore_index=True)

        # Guardar resumen global
        df_total.to_csv(ruta_subcarpeta / "resumen_global.csv", index=False)

procesar_datos()