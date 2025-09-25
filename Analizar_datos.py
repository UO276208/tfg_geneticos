import pandas as pd
from pathlib import Path

def calcular_convergencia(path_medias, epsilon=0.01):
    """
    Calcula en qué generación el fitness medio se estabiliza.
    Estabilización = fitness medio dentro de (1+ε) del valor final.
    """
    df = pd.read_csv(path_medias)
    fitness_final = df["Fitness_medio"].iloc[-1]
    umbral = fitness_final * (1 + epsilon)

    # Primer índice (generación) donde se cumple la condición
    convergencia_gen = df.index[
        (df["Fitness_medio"] <= umbral)
    ].min()

    if pd.isna(convergencia_gen):
        convergencia_gen = len(df) - 1  # nunca estabilizó

    return int(convergencia_gen), len(df)


def analizar_datos():
    ruta_base = Path(Path(__file__).resolve().parent, 'SSGAs', 'data_copia')
    resumen_global = ruta_base / "resumen_global.csv"

    if not resumen_global.exists():
        raise FileNotFoundError(f"No se encontró {resumen_global}, ejecuta primero el script de procesado.")

    # Leer resumen global
    df = pd.read_csv(resumen_global)

    # -----------------------------
    # 1. Añadir métricas derivadas
    # -----------------------------
    # Porcentaje de mejora
    df["mejora_pct"] = (df["fitness_ini"] - df["fitness_medio"]) / df["fitness_ini"] * 100

    # Score combinado (fitness * tiempo)
    df["score"] = df["fitness_max"] * df["tiempo"]

    # Convergencia: leer cada archivo de medias
    convergencias = []
    for _, row in df.iterrows():
        ruta_medias = ruta_base / row["subcarpeta"] / row["subsubcarpeta"] / "medias" / row["archivo"]
        if ruta_medias.exists():
            g_conv, g_total = calcular_convergencia(ruta_medias)
            conv_pct = g_conv / g_total * 100
            convergencias.append((g_conv, g_total, conv_pct))
        else:
            convergencias.append((None, None, None))

    df["gen_convergencia"], df["gen_total"], df["convergencia_pct"] = zip(*convergencias)

    # Guardar DF con métricas añadidas
    df.to_csv(ruta_base / "resumen_global_extendido.csv", index=False)

    # -----------------------------
    # 2. Métricas por subcarpeta
    # -----------------------------
    agg_sub = df.groupby("subcarpeta").agg({
        "fitness_max": ["mean", "min", "std"],
        "fitness_medio": "mean",
        "tiempo": ["mean", "sum"],
        "mejora_pct": "mean",
        "convergencia_pct": "mean"
    }).reset_index()

    agg_sub.columns = [
        "subcarpeta",
        "fitness_max_mean", "fitness_max_min", "fitness_max_std",
        "fitness_medio_mean",
        "tiempo_mean", "tiempo_total",
        "mejora_pct_mean",
        "convergencia_pct_mean"
    ]

    agg_sub["score"] = agg_sub["fitness_max_mean"] * agg_sub["tiempo_mean"]
    agg_sub.to_csv(ruta_base / "analisis_por_subcarpeta.csv", index=False)

    # -----------------------------
    # 3. Métricas por subsubcarpeta
    # -----------------------------
    agg_subsub = df.groupby(["subcarpeta", "subsubcarpeta"]).agg({
        "fitness_max": ["mean", "min"],
        "fitness_medio": "mean",
        "tiempo": ["mean", "sum"],
        "mejora_pct": "mean",
        "convergencia_pct": "mean"
    }).reset_index()

    agg_subsub.columns = [
        "subcarpeta", "subsubcarpeta",
        "fitness_max_mean", "fitness_max_min",
        "fitness_medio_mean",
        "tiempo_mean", "tiempo_total",
        "mejora_pct_mean",
        "convergencia_pct_mean"
    ]

    agg_subsub["score"] = agg_subsub["fitness_max_mean"] * agg_subsub["tiempo_mean"]
    agg_subsub.to_csv(ruta_base / "analisis_por_subsubcarpeta.csv", index=False)

    # -----------------------------
    # 4. Mejores resultados globales
    # -----------------------------
    mejor_fitness = df.loc[df["fitness_max"].idxmin()]
    mejor_score = df.loc[df["score"].idxmin()]
    mejor_mejora = df.loc[df["mejora_pct"].idxmax()]

    df_mejores = pd.DataFrame([
        {
            "criterio": "fitness_min",
            "subcarpeta": mejor_fitness["subcarpeta"],
            "subsubcarpeta": mejor_fitness["subsubcarpeta"],
            "archivo": mejor_fitness["archivo"],
            "fitness_max": mejor_fitness["fitness_max"],
            "fitness_medio": mejor_fitness["fitness_medio"],
            "tiempo": mejor_fitness["tiempo"],
            "mejora_pct": mejor_fitness["mejora_pct"],
            "convergencia_pct": mejor_fitness["convergencia_pct"]
        },
        {
            "criterio": "score_min",
            "subcarpeta": mejor_score["subcarpeta"],
            "subsubcarpeta": mejor_score["subsubcarpeta"],
            "archivo": mejor_score["archivo"],
            "fitness_max": mejor_score["fitness_max"],
            "fitness_medio": mejor_score["fitness_medio"],
            "tiempo": mejor_score["tiempo"],
            "mejora_pct": mejor_score["mejora_pct"],
            "convergencia_pct": mejor_score["convergencia_pct"]
        },
        {
            "criterio": "mejora_max",
            "subcarpeta": mejor_mejora["subcarpeta"],
            "subsubcarpeta": mejor_mejora["subsubcarpeta"],
            "archivo": mejor_mejora["archivo"],
            "fitness_max": mejor_mejora["fitness_max"],
            "fitness_medio": mejor_mejora["fitness_medio"],
            "tiempo": mejor_mejora["tiempo"],
            "mejora_pct": mejor_mejora["mejora_pct"],
            "convergencia_pct": mejor_mejora["convergencia_pct"]
        }
    ])

    df_mejores.to_csv(ruta_base / "mejores_globales.csv", index=False)


if __name__ == "__main__":
    analizar_datos()
