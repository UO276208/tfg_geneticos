import pandas as pd
from pathlib import Path

def ranking_ejecuciones(ruta_subcarpeta: Path):
    # Buscar todos los CSV de la carpeta
    archivos_csv = sorted(ruta_subcarpeta.glob("*.csv"))
    if not archivos_csv:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {ruta_subcarpeta}")

    resultados = []

    for archivo in archivos_csv:
        df = pd.read_csv(archivo)
        fitness_final = df.iloc[-1]["Fitness_max"]  # Última generación
        resultados.append((archivo.name, fitness_final))

    # Ordenar por fitness_final (menor es mejor)
    resultados.sort(key=lambda x: x[1])

    # Mostrar ranking
    print("\n📊 Ranking de ejecuciones por Fitness_max final:")
    for i, (nombre, fitness) in enumerate(resultados, start=1):
        print(f"{i:02d}. {nombre} --> Fitness_max final: {fitness:.4f}")

    return resultados

# Ejemplo de uso
ruta_subcarpeta = Path(Path(__file__).resolve().parent.parent, 'SSGAs', 'data2', 'graficas', 'medias')
ranking = ranking_ejecuciones(ruta_subcarpeta)