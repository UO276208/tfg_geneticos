import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_BASE = Path(__file__).resolve().parent.parent.parent / "SSGAs" / "data_copia"
RUTA_SALIDA = Path(__file__).resolve().parent / "resultados" / "mejor_configuracion"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# ============================================================
# FUNCIÓN DE ANÁLISIS
# ============================================================

def analizar_archivo(ruta_csv, variante):
    """
    Procesa un archivo de medias y devuelve métricas:
      - Mejor fitness (mínimo coste)
      - Generación donde se alcanza
      - Tiempo total y tiempo hasta la mejor solución
    En variantes con penalización, usa solo Fitness_max_factible.
    """
    try:
        df = pd.read_csv(ruta_csv)
    except Exception as e:
        print(f"⚠️ Error leyendo {ruta_csv.name}: {e}")
        return None

    if "Generacion" not in df.columns or "Tiempo" not in df.columns:
        print(f"⚠️ Archivo sin columnas esperadas: {ruta_csv.name}")
        return None

    # Agrupar por generación (por si hay varias filas por generación)
    df = df.groupby("Generacion", as_index=False).mean(numeric_only=True)

    # Selección del fitness según el tipo de variante
    if "penalizaciones" in variante and "Fitness_max_factible" in df.columns:
        columna_fitness = "Fitness_max_factible"
    elif "Fitness_max" in df.columns:
        columna_fitness = "Fitness_max"
    else:
        print(f"⚠️ No hay columnas de fitness adecuadas en {ruta_csv.name}")
        return None

    # Quitar valores no válidos
    df[columna_fitness] = pd.to_numeric(df[columna_fitness], errors="coerce")
    df = df.dropna(subset=[columna_fitness])

    if df.empty:
        # No hay soluciones factibles
        return {
            "Variante": variante,
            "Archivo": ruta_csv.name,
            "Generacion_mejor": np.nan,
            "Fitness_mejor": np.nan,
            "Tiempo_total": np.nan,
            "Tiempo_mejor": np.nan,
        }

    # Buscar el mejor fitness (mínimo coste)
    mejor_idx = df[columna_fitness].idxmin()
    mejor_fitness = df.loc[mejor_idx, columna_fitness]
    gen_mejor = df.loc[mejor_idx, "Generacion"]

    # Tiempos
    gen_final = df["Generacion"].max()
    tiempo_total = df["Tiempo"].iloc[-1]

    # Tiempo estimado para alcanzar la mejor solución (regla de tres)
    if pd.notna(gen_mejor) and gen_final > 0:
        tiempo_mejor = (tiempo_total * gen_mejor) / gen_final
    else:
        tiempo_mejor = np.nan

    return {
        "Variante": variante,
        "Archivo": ruta_csv.name,
        "Generacion_mejor": int(gen_mejor),
        "Fitness_mejor": float(mejor_fitness),
        "Tiempo_total": float(tiempo_total),
        "Tiempo_mejor": float(tiempo_mejor),
    }

# ============================================================
# RECORRIDO DE DIRECTORIOS
# ============================================================

resultados = []

for variante_dir in RUTA_BASE.iterdir():
    if not variante_dir.is_dir():
        continue

    variante = variante_dir.name
    print(f"🔍 Analizando variante: {variante}")

    for experimento_dir in variante_dir.iterdir():
        if not experimento_dir.is_dir() or experimento_dir.name == "gen":
            continue  # ignorar tests de convergencia (100k generaciones)

        carpeta_medias = experimento_dir / "medias"
        if not carpeta_medias.exists():
            continue

        for archivo in carpeta_medias.glob("*.csv"):
            res = analizar_archivo(archivo, variante)
            if res is not None:
                resultados.append(res)

# ============================================================
# SELECCIÓN DE LA MEJOR CONFIGURACIÓN POR VARIANTE
# ============================================================

df_resultados = pd.DataFrame(resultados)

# Si alguna variante no tiene soluciones factibles, se mostrará como NaN
df_resultados_sorted = df_resultados.sort_values(
    by=["Fitness_mejor", "Generacion_mejor", "Tiempo_total"],
    ascending=[True, True, True],
    na_position="last"
)

mejores_por_variante = (
    df_resultados_sorted.groupby("Variante", dropna=False).first().reset_index()
)

# ============================================================
# EXPORTACIÓN DE RESULTADOS
# ============================================================

salida_csv = RUTA_SALIDA / "mejores_configuraciones.csv"
mejores_por_variante.to_csv(salida_csv, index=False)

print("✅ Resultados guardados en:", salida_csv)
print(mejores_por_variante)
