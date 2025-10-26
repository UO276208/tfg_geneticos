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
        df_sorted = grupo.sort_values(param).reset_index(drop=True)

        # ----------------------------------------------------------------
        # 🔸 CASO 1: Parámetro gamma → usar salto configurable (0.2, 0.5, etc.)
        # ----------------------------------------------------------------
        if experimento == "gamma":  # <- cuidado: debe coincidir con la clave en df_resumen, no "Gamma"
            salto_gamma = 0.2  # cambia a 0.2, 0.5, etc.
            tolerancia = 1e-6
            pares_validos = []

            # Compara todos los pares posibles (no solo consecutivos)
            for i in range(len(df_sorted)):
                for j in range(i + 1, len(df_sorted)):
                    delta = abs(df_sorted.loc[j, param] - df_sorted.loc[i, param])
                    if np.isclose(delta, salto_gamma, atol=tolerancia):
                        pares_validos.append((i, j))

            delta_fit_pct = []
            for i, j in pares_validos:
                F_i = df_sorted.loc[i, "Fitness_medio_final_mean"]
                F_j = df_sorted.loc[j, "Fitness_medio_final_mean"]
                delta_fit_pct.append(abs((F_j - F_i) / F_i) * 100)

            impacto_medio = np.mean(delta_fit_pct) if delta_fit_pct else np.nan
            impacto_normalizado = np.nan  # no se normaliza

        # ----------------------------------------------------------------
        # 🔸 CASO 2: Resto de parámetros → saltos consecutivos (original)
        # ----------------------------------------------------------------
        else:
            df_sorted["delta_fit_pct"] = (
                                                 df_sorted["Fitness_medio_final_mean"].diff().abs()
                                                 / df_sorted["Fitness_medio_final_mean"].shift(1)
                                         ) * 100
            impacto_medio = df_sorted["delta_fit_pct"].mean()

            # Cálculo del salto medio solo como referencia
            delta_param_mean = df_sorted[param].diff().abs().mean()
            impacto_normalizado = (
                impacto_medio / delta_param_mean
                if pd.notna(delta_param_mean) and delta_param_mean > 0
                else np.nan
            )

        # ----------------------------------------------------------------
        # 🔸 Guardar resultados
        # ----------------------------------------------------------------
        resultados_impacto.append({
            "Variante": variante,
            "Parametro": param,
            "Experimento": experimento,
            "Impacto_medio_pct": impacto_medio,
            "Impacto_normalizado": impacto_normalizado
        })

    # --------------------------------------------------------------------
    # 🔹 Exportar resultados
    # --------------------------------------------------------------------
    df_impacto = pd.DataFrame(resultados_impacto)
    df_impacto.to_csv(ruta_salida / "impacto_parametros.csv", index=False)
    print("✅ Archivo de impacto medio generado (γ con salto 0.2).")

# Ejecutar
if __name__ == "__main__":
    procesar_datos()
