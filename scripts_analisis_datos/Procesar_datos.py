from pathlib import Path
import pandas as pd
import re


# Función auxiliar: parsear parámetros desde el nombre del archivo
def extraer_parametros(nombre_archivo):
    """
    Extrae pop, gen, mut y gamma de un nombre de archivo con formato:
    ...P{pop}_G{gen}_{mut}-gamma_{gamma}...
    """
    pat = re.compile(r"P(\d+)_G(\d+)_(\d+\.\d+)-gamma_(\d+\.\d+)")
    m = pat.search(nombre_archivo)
    if not m:
        return None, None, None, None
    pop, gen, mut, gamma = m.groups()
    return int(pop), int(gen), float(mut), float(gamma)


# Función auxiliar: detectar algoritmo y penalizaciones
def extraer_info_algoritmo(nombre_archivo):
    if nombre_archivo.startswith("prim"):
        algoritmo = "prim"
    elif nombre_archivo.startswith("kruskal"):
        algoritmo = "kruskal"
    else:
        algoritmo = "desconocido"

    penalizaciones = "con_penalizaciones" if "penalizaciones" in nombre_archivo else "sin_penalizaciones"
    return algoritmo, penalizaciones


#Calcula el coste medio por generacion y lo guarda
def crear_medias(ruta_subcarpeta, nombre_archivo):

    ruta_archivo = Path(ruta_subcarpeta, nombre_archivo)
    ruta_salida = Path(ruta_subcarpeta, 'medias')
    ruta_salida.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ruta_archivo)

    # Comprobación de columnas esperadas
    columnas_necesarias = {'Generacion', 'Fitness_max', 'Fitness_medio', 'Tiempo'}
    if not columnas_necesarias.issubset(df.columns):
        raise ValueError(
            f"El archivo {ruta_archivo} no contiene las columnas requeridas "
            f"{columnas_necesarias}."
        )

    # Definir columnas de agregación básicas
    columnas_agrupar = {
        'Fitness_max': 'mean',
        'Fitness_medio': 'mean',
        'Tiempo': 'mean'
    }

    # Añadir columnas extra solo si existen (archivos con penalizaciones)
    if 'Fitness_max_factible' in df.columns:
        df['Fitness_max_factible'] = df['Fitness_max_factible'].replace(-1, pd.NA)
        columnas_agrupar['Fitness_max_factible'] = 'mean'
    if 'N_violaciones' in df.columns:
        columnas_agrupar['N_violaciones'] = 'mean'

    # Calcular medias por generación
    df_grouped_mean = df.groupby('Generacion', as_index=False).agg(columnas_agrupar)

    # Guardar archivo con todas las medias
    df_grouped_mean.to_csv(Path(ruta_salida, nombre_archivo), index=False)

    # Fila primera y última (para los resúmenes)
    primera_gen = df_grouped_mean.iloc[0]
    ultima_gen = df_grouped_mean.iloc[-1]

    # Extraer parámetros desde el nombre
    pop, gen, mut, gamma = extraer_parametros(nombre_archivo)

    # Extraer algoritmo y penalizaciones
    algoritmo, penalizaciones = extraer_info_algoritmo(nombre_archivo)

    return pd.DataFrame([{
        "archivo": nombre_archivo,
        "fitness_ini": primera_gen["Fitness_medio"],
        "fitness_max": ultima_gen["Fitness_max"],
        "fitness_medio": ultima_gen["Fitness_medio"],
        "tiempo": ultima_gen["Tiempo"],
        "pop": pop,
        "gen": gen,
        "mut": mut,
        "gamma": gamma,
        "algoritmo": algoritmo,
        "penalizaciones": penalizaciones
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

    return pd.DataFrame(columns=[
        "archivo", "fitness_ini", "fitness_max", "fitness_medio", "tiempo",
        "pop", "gen", "mut", "gamma", "algoritmo", "penalizaciones"
    ])

def procesar_datos():
    ruta_subcarpeta = Path(Path(__file__).resolve().parent.parent, 'SSGAs', 'data')
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