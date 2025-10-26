import pandas as pd
from pathlib import Path

def analizar_datos():
    ruta_base = Path(Path(__file__).resolve().parent.parent.parent, "../../SSGAs", "data_copia")
    resumen_global = ruta_base / "resumen_global.csv"

    if not resumen_global.exists():
        raise FileNotFoundError("No se encontró resumen_global.csv. Ejecuta primero el script de procesado.")

    df = pd.read_csv(resumen_global)

    # Calcular porcentaje de mejora
    df["mejora_abs"] = df["fitness_ini"] - df["fitness_medio"]
    df["mejora_pct"] = (df["mejora_abs"] / df["fitness_ini"]) * 100

    # Guardar análisis general (sin convergencia)
    salida = ruta_base / "analisis.csv"
    df.to_csv(salida, index=False)
    print(f"Análisis general guardado en {salida}")

    # === Mejor configuración por subcarpeta (incluyendo empates) ===
    mejores_configuraciones = []

    for subcarpeta, grupo in df.groupby("subcarpeta"):
        # Mejor fitness encontrado en esta subcarpeta
        mejor_fitness = grupo["fitness_max"].min()
        candidatos = grupo[grupo["fitness_max"] == mejor_fitness]

        # Entre ellos, también miramos el mejor score (fitness*tiempo)
        mejor_score = (candidatos["fitness_max"] * candidatos["tiempo"]).min()
        empatados = candidatos[candidatos["fitness_max"] * candidatos["tiempo"] == mejor_score]

        for _, row in empatados.iterrows():
            mejores_configuraciones.append({
                "subcarpeta": row["subcarpeta"],
                "subsubcarpeta": row["subsubcarpeta"],
                "archivo": row["archivo"],
                "fitness_max": row["fitness_max"],
                "tiempo": row["tiempo"],
                "empate": len(empatados) > 1  # True si hubo empate
            })

    df_mejores_config = pd.DataFrame(mejores_configuraciones)

    ruta_mejores = ruta_base / "mejores_configuraciones.csv"
    df_mejores_config.to_csv(ruta_mejores, index=False)

    print(f"Guardado mejores configuraciones en {ruta_mejores}")



if __name__ == "__main__":
    analizar_datos()
