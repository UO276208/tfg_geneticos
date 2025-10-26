import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_BASE = Path(__file__).resolve().parent.parent.parent / "SSGAs" / "data"
RUTA_SALIDA = Path(__file__).resolve().parent / "graficas_generadas" / "grafica_variantes"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# ============================================================
# ARCHIVOS DE LAS MEJORES CONFIGURACIONES (medias)
# ============================================================

archivos_mejores = {
    "K-Prim (sin pen.)": RUTA_BASE / "hard_prim" / "gamma" / "medias" / "prim_imp_h_bayg29P2000_G20000_0.03-gamma_0.1.csv",
    "K-Kruskal (sin pen.)": RUTA_BASE / "hard_kruskal" / "gamma" / "medias" / "kruskal_imp_h_bayg29_gammatestP2000_G20000_0.03-gamma_2.5.csv",
    "Prim (con pen.)": RUTA_BASE / "penalizaciones_prim" / "gamma" / "medias" / "prim_imp_penalizaciones_bayg29_gammatestP2000_G20000_0.03-gamma_0.1.csv",
    "Kruskal (con pen.)": RUTA_BASE / "penalizaciones_kruskal" / "gamma" / "medias" / "kruskal_imp_penalizaciones_bayg29_gammatestP2000_G20000_0.03-gamma_0.0.csv"
}

COLORES = {
    "K-Prim (sin pen.)": "tab:blue",
    "K-Kruskal (sin pen.)": "tab:orange",
    "Prim (con pen.)": "tab:green",
    "Kruskal (con pen.)": "tab:red"
}

# ============================================================
# FUNCIÓN AUXILIAR PARA CARGAR DATOS
# ============================================================

def cargar_datos_fitness(ruta, nombre):
    """Devuelve (generaciones, fitness_y) según si hay penalizaciones o no."""
    if not ruta.exists():
        print(f"⚠️ Archivo no encontrado para {nombre}: {ruta}")
        return None, None

    df = pd.read_csv(ruta)
    df = df.groupby("Generacion", as_index=False).mean(numeric_only=True)

    if "penalizaciones" in nombre.lower() and "Fitness_max_factible" in df.columns:
        y = df["Fitness_max_factible"]
    elif "Fitness_max" in df.columns:
        y = df["Fitness_max"]
    else:
        print(f"⚠️ Sin datos factibles para {nombre}")
        return None, None

    if y.isna().all():
        print(f"⚠️ Todos los valores NaN para {nombre}")
        return None, None

    return df["Generacion"], y


# ============================================================
# GRÁFICA 1: TODAS LAS VARIANTES
# ============================================================

plt.figure(figsize=(8, 5))

for nombre, ruta in archivos_mejores.items():
    generaciones, y = cargar_datos_fitness(ruta, nombre)
    if generaciones is None:
        continue
    plt.plot(generaciones, y, color=COLORES[nombre], label=nombre)

plt.xlabel("Generación")
plt.ylabel("Fitness (coste)")
plt.title("Comparativa de convergencia entre variantes (fitness máximo)")
plt.legend(frameon=False, fontsize=8)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RUTA_SALIDA / "comparativa_convergencia_todas.png", dpi=300)
plt.close()

# ============================================================
# GRÁFICA 2: SOLO VARIANTES BASADAS EN PRIM
# ============================================================

plt.figure(figsize=(8, 5))

for nombre in ["K-Prim (sin pen.)", "Prim (con pen.)"]:
    generaciones, y = cargar_datos_fitness(archivos_mejores[nombre], nombre)
    if generaciones is None:
        continue
    plt.plot(generaciones, y, color=COLORES[nombre], label=nombre)

plt.xlabel("Generación")
plt.ylabel("Fitness (coste)")
plt.title("Comparativa de convergencia – variantes basadas en Prim")
plt.legend(frameon=False, fontsize=9)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RUTA_SALIDA / "comparativa_convergencia_prim.png", dpi=300)
plt.close()

print("✅ Gráficas generadas en:", RUTA_SALIDA)
