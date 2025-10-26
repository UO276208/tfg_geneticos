import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_BASE = Path(__file__).resolve().parent.parent.parent / "SSGAs" / "data_copia"
RUTA_SALIDA = Path(__file__).resolve().parent / "resultados" / "resultados_convergencia"
RUTA_SALIDA.mkdir(exist_ok=True)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def detectar_convergencia(df):
    """
    Detecta la primera generación a partir de la cual el fitness medio deja de variar.
    Se considera convergencia cuando el fitness medio se mantiene constante
    (dentro de una tolerancia numérica) hasta el final de la ejecución.
    """
    df = df.copy()
    df["delta_fit"] = df["Fitness_medio"].diff()
    tol = 1e-8  # tolerancia por redondeo numérico

    for i in range(1, len(df)):
        tramo = df["delta_fit"].iloc[i:]
        if np.all(np.abs(tramo) < tol):
            return int(df.loc[i, "Generacion"])
    return np.nan


def procesar_archivo(ruta_csv, variante):
    """
    Calcula métricas de convergencia para un archivo de datos medios.
    Usa el fitness medio para el análisis de convergencia y mejora,
    pero el fitness final corresponde al mejor individuo factible
    si se trata de una variante con penalizaciones.
    """
    df = pd.read_csv(ruta_csv)
    df = df.groupby("Generacion", as_index=False).mean(numeric_only=True)

    gen_conv = detectar_convergencia(df)
    fit_ini = df["Fitness_medio"].iloc[0]
    fit_fin_medio = df["Fitness_medio"].iloc[-1]

    # --- Fitness final para reporte ---
    if "penalizaciones" in variante and "Fitness_max_factible" in df.columns:
        factibles = df["Fitness_max_factible"].dropna()
        fit_final_reporte = factibles.iloc[-1] if not factibles.empty else np.nan
    else:
        fit_final_reporte = fit_fin_medio

    # --- Métricas de mejora (usando fitness medio) ---
    mejora_abs = fit_fin_medio - fit_ini
    mejora_pct = (mejora_abs / fit_ini) * 100
    mejora_media = mejora_abs / len(df)

    # ------------------------------------------------------------
    # Generación de mejor solución (G_best)
    # ------------------------------------------------------------
    if "penalizaciones" in variante and "Fitness_max_factible" in df.columns:
        factibles = df.dropna(subset=["Fitness_max_factible"])
        if not factibles.empty:
            idx_best = factibles["Fitness_max_factible"].idxmin()
            g_best = int(factibles.loc[idx_best, "Generacion"])
        else:
            g_best = np.nan
    else:
        if "Fitness_max" in df.columns:
            idx_best = df["Fitness_max"].idxmin()
            g_best = int(df.loc[idx_best, "Generacion"])
        else:
            g_best = np.nan

    return {
        "Variante": variante,
        "Archivo": ruta_csv.name,
        "Generacion_convergencia": gen_conv,
        "Generacion_mejor_solucion": g_best,  # 👈 NUEVA MÉTRICA
        "Fitness_inicial": fit_ini,
        "Fitness_final": fit_final_reporte,
        "Mejora_total": mejora_abs,
        "Mejora_pct": mejora_pct,
        "Mejora_media_por_gen": mejora_media
    }

# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

resultados = []

for sub in RUTA_BASE.iterdir():
    if not sub.is_dir():
        continue

    variante = sub.name
    carpeta_gen = sub / "gen"
    if not carpeta_gen.exists():
        continue

    print(f"Analizando convergencia en {variante}...")

    for archivo in carpeta_gen.glob("*.csv"):
        try:
            res = procesar_archivo(archivo, variante)
            resultados.append(res)
        except Exception as e:
            print(f"⚠️ Error procesando {archivo.name}: {e}")

# ============================================================
# EXPORTACIÓN DE RESULTADOS
# ============================================================

df_resultados = pd.DataFrame(resultados)
salida_csv = RUTA_SALIDA / "resumen_convergencia.csv"
df_resultados.to_csv(salida_csv, index=False)

print(f"✅ Resultados de convergencia guardados en: {salida_csv}")
