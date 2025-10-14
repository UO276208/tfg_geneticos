import pandas as pd
import numpy as np
import re
from pathlib import Path

# --- Función auxiliar: extraer parámetros del nombre ---
def extraer_parametros(nombre_archivo):
    pat = re.compile(r"P(\d+)_G(\d+)_(\d+\.\d+)-gamma_(\d+\.\d+)")
    m = pat.search(nombre_archivo)
    if not m:
        return None, None, None, None
    pop, gen, mut, gamma = m.groups()
    return int(pop), int(gen), float(mut), float(gamma)

# --- Procesar datos experimentales ---
def procesar_datos():
    ruta_base = Path(__file__).resolve().parent / "SSGAs" / "data_copia"
    ruta_salida = ruta_base / "procesados"
    ruta_salida.mkdir(parents=True, exist_ok=True)

    resumenes = []
    evoluciones = []

    # Recorrer variantes (subcarpetas principales)
    for carpeta_variante in ruta_base.iterdir():
        if not carpeta_variante.is_dir():
            continue
        variante = carpeta_variante.name

        for subcarpeta in carpeta_variante.iterdir():
            if not subcarpeta.is_dir():
                continue
            experimento = subcarpeta.name

            for archivo in subcarpeta.glob("*.csv"):
                pop, gen, mut, gamma = extraer_parametros(archivo.name)
                if pop is None:
                    continue

                df = pd.read_csv(archivo)
                df["Variante"] = variante
                df["Experimento"] = experimento
                df["Archivo"] = archivo.name
                df["Poblacion"] = pop
                df["Generaciones"] = gen
                df["Prob_mutacion"] = mut
                df["Gamma"] = gamma

                # --- Evolución promedio por generación ---
                df_media = (
                    df.groupby("Generacion")
                      .mean(numeric_only=True)
                      .reset_index()
                )
                df_media["Variante"] = variante
                df_media["Experimento"] = experimento
                df_media["Poblacion"] = pop
                df_media["Generaciones"] = gen
                df_media["Prob_mutacion"] = mut
                df_media["Gamma"] = gamma
                evoluciones.append(df_media)

                # --- Resumen final por configuración ---
                fila = {
                    "Variante": variante,
                    "Experimento": experimento,
                    "Configuracion": archivo.name,
                    "Poblacion": pop,
                    "Generaciones": gen,
                    "Prob_mutacion": mut,
                    "Gamma": gamma,
                    "n_ejecuciones": df["Fuente"].nunique() if "Fuente" in df.columns else np.nan,
                    "Fitness_max_final_mean": df.query("Generacion == Generacion.max()")["Fitness_max"].mean(),
                    "Fitness_max_final_std": df.query("Generacion == Generacion.max()")["Fitness_max"].std(),
                    "Fitness_medio_final_mean": df.query("Generacion == Generacion.max()")["Fitness_medio"].mean(),
                    "Fitness_medio_final_std": df.query("Generacion == Generacion.max()")["Fitness_medio"].std(),
                    "Tiempo_total_mean": df.groupby("Fuente")["Tiempo"].sum().mean() if "Fuente" in df.columns else np.nan,
                    "Tiempo_total_std": df.groupby("Fuente")["Tiempo"].sum().std() if "Fuente" in df.columns else np.nan,
                }

                # --- Campos extra para versiones con penalización ---
                if "Fitness_max_factible" in df.columns:
                    fila["Fitness_max_factible_final_mean"] = (
                        df.query("Generacion == Generacion.max()")["Fitness_max_factible"].mean()
                    )
                if "N_violaciones" in df.columns:
                    fila["N_violaciones_final_mean"] = (
                        df.query("Generacion == Generacion.max()")["N_violaciones"].mean()
                    )

                resumenes.append(fila)

    # --- Guardar resumen y evolución ---
    df_resumen = pd.DataFrame(resumenes)
    df_evolucion = pd.concat(evoluciones, ignore_index=True)
    df_resumen.to_csv(ruta_salida / "resumen_configuraciones.csv", index=False)
    df_evolucion.to_csv(ruta_salida / "evolucion_promedio.csv", index=False)

    print("✅ Archivos resumen generados.")

    # --------------------------------------------------------------------
    # 🔹 Cálculo automático del impacto medio de cada parámetro
    # --------------------------------------------------------------------
    resultados_impacto = []
    parametros = {
        "gamma": "Gamma",
        "mut": "Prob_mutacion",
        "pop": "Poblacion",
        "gen": "Generaciones"
    }

    for (variante, experimento), grupo in df_resumen.groupby(["Variante", "Experimento"]):
        if experimento not in parametros:
            continue
        param = parametros[experimento]
        df_sorted = grupo.sort_values(param)

        # Diferencia relativa de fitness entre configuraciones consecutivas
        df_sorted["delta_fit_pct"] = (
            df_sorted["Fitness_medio_final_mean"].diff().abs()
            / df_sorted["Fitness_medio_final_mean"].shift(1)
        ) * 100

        # Promedio ignorando el primer NaN
        impacto_medio = df_sorted["delta_fit_pct"].mean()

        # Normalizado por el salto medio del parámetro (por si los pasos no son iguales)
        delta_param_mean = df_sorted[param].diff().abs().mean()
        if pd.notna(delta_param_mean) and delta_param_mean > 0:
            impacto_normalizado = impacto_medio / delta_param_mean
        else:
            impacto_normalizado = np.nan

        resultados_impacto.append({
            "Variante": variante,
            "Parametro": param,
            "Experimento": experimento,
            "Impacto_medio_pct": impacto_medio,
            "Impacto_normalizado": impacto_normalizado
        })

    df_impacto = pd.DataFrame(resultados_impacto)
    df_impacto.to_csv(ruta_salida / "impacto_parametros.csv", index=False)
    print("✅ Archivo de impacto medio generado.")

# Ejecutar
if __name__ == "__main__":
    procesar_datos()
