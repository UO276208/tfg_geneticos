"""
visualizacion_resultados.py
Genera las gráficas principales a partir de los resultados procesados.
Requiere: matplotlib, seaborn, pandas
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# 🧭 Rutas de entrada
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "SSGAs" / "data_copia"
RESULTADOS_DIR = DATA_DIR / "procesados"
SALIDA_DIR = RESULTADOS_DIR / "graficas"
SALIDA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# 📥 Cargar datos resumen
# ---------------------------------------------------------------------
df_impacto = pd.read_csv(RESULTADOS_DIR / "impacto_parametros.csv")

# ---------------------------------------------------------------------
# 🎨 1. Gráfico de barras — impacto medio porcentual
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df_impacto, x="Parametro", y="Impacto_medio_pct", hue="Variante")

plt.ylabel("Impacto medio porcentual (%)")
plt.xlabel("Parámetro")
plt.legend(title="Variante", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(SALIDA_DIR / "impacto_parametros_barras.png", dpi=300)
plt.close()
print("✅ Gráfico de barras generado.")

# ---------------------------------------------------------------------
# 📈 2. Curvas de evolución del fitness medio por generación
# ---------------------------------------------------------------------
def graficar_convergencia(variante: str, experimento: str):
    """Grafica fitness medio para valores extremos del parámetro."""
    ruta_medias = DATA_DIR / variante / experimento / "medias"
    if not ruta_medias.exists():
        return

    csv_files = sorted([
        f for f in ruta_medias.glob("*.csv")
        if "resumen" not in f.name.lower()
    ])
    if len(csv_files) < 2:
        return

    df_low = pd.read_csv(csv_files[0])
    df_high = pd.read_csv(csv_files[-1])

    plt.figure(figsize=(8, 5))
    plt.plot(df_low["Generacion"], df_low["Fitness_medio"], label=f"{csv_files[0].stem}")
    plt.plot(df_high["Generacion"], df_high["Fitness_medio"], label=f"{csv_files[-1].stem}")

    plt.xlabel("Generación")
    plt.ylabel("Fitness medio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SALIDA_DIR / f"convergencia_{variante}_{experimento}.png", dpi=300)
    plt.close()

for variante in ["hard_prim", "hard_kruskal", "penalizaciones_prim", "penalizaciones_kruskal"]:
    for experimento in ["gamma", "mut", "pop", "limitgentest"]:
        graficar_convergencia(variante, experimento)

print("✅ Gráficas de convergencia generadas.")

# ---------------------------------------------------------------------
# 📦 3. Boxplots del fitness final por parámetro y variante
# ---------------------------------------------------------------------
# Para esto usaremos los CSVs originales de medias (última generación)
def extraer_fitness_final(variante: str, experimento: str):
    ruta_medias = DATA_DIR / variante / experimento / "medias"
    if not ruta_medias.exists():
        return pd.DataFrame()

    datos = []
    for csv in ruta_medias.glob("*.csv"):
        df = pd.read_csv(csv)
        if "Fitness_medio" in df.columns:
            fitness_final = df["Fitness_medio"].iloc[-1]
            datos.append({"Variante": variante, "Experimento": experimento,
                          "Archivo": csv.stem, "Fitness_final": fitness_final})
    return pd.DataFrame(datos)

df_box = pd.concat(
    [extraer_fitness_final(v, e)
     for v in ["hard_prim", "hard_kruskal", "penalizaciones_prim", "penalizaciones_kruskal"]
     for e in ["gamma", "mut", "pop", "limitgentest"]],
    ignore_index=True
)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_box, x="Experimento", y="Fitness_final", hue="Variante")
plt.xlabel("Parámetro estudiado")
plt.ylabel("Fitness final")
plt.legend(title="Variante", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(SALIDA_DIR / "boxplot_fitness_final.png", dpi=300)
plt.close()
print("✅ Boxplots generados.")

print("\n🎉 Todas las gráficas se han creado en:", SALIDA_DIR)
