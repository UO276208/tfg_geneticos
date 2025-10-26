import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_BASE = Path(__file__).resolve().parent.parent.parent / "SSGAs" / "data_copia"
RUTA_CONVERGENCIA = Path(__file__).resolve().parent.parent / "analisis" / "resultados" / "resultados_convergencia" / "resumen_convergencia.csv"

# 🔸 Nueva ruta de salida
RUTA_SALIDA = Path(__file__).resolve().parent / "graficas_generadas" / "graficas_convergencia"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# ============================================================
# CARGAR DATOS DE CONVERGENCIA
# ============================================================

df_conv = pd.read_csv(RUTA_CONVERGENCIA)

# ============================================================
# GENERACIÓN DE GRÁFICAS DETALLADAS
# ============================================================

for sub in RUTA_BASE.iterdir():
    if not sub.is_dir():
        continue

    variante = sub.name
    carpeta_gen = sub / "gen"
    if not carpeta_gen.exists():
        continue

    for archivo in carpeta_gen.glob("*.csv"):
        try:
            df = pd.read_csv(archivo)
            df = df.groupby("Generacion", as_index=False).mean(numeric_only=True)

            # Buscar generación de convergencia
            fila_conv = df_conv[
                (df_conv["Variante"] == variante) &
                (df_conv["Archivo"] == archivo.name)
            ]
            gen_conv = None
            gen_best = None

            if not fila_conv.empty:
                gen_conv = fila_conv["Generacion_convergencia"].values[0]
                if pd.isna(gen_conv):
                    gen_conv = None

                    # Generación donde se encontró el mejor individuo
            if "Generacion_mejor_solucion" in fila_conv.columns:
                gen_best = fila_conv["Generacion_mejor_solucion"].values[0]
                if pd.isna(gen_best):
                    gen_best = None

            # Crear figura
            plt.figure(figsize=(7, 4))

            # 🔹 Línea negra: fitness medio
            plt.plot(df["Generacion"], df["Fitness_medio"], color="black", label="Fitness medio")

            # 🔹 Líneas de fitness máximo
            if "penalizaciones" in variante:
                if "Fitness_max" in df.columns:
                    plt.plot(df["Generacion"], df["Fitness_max"],
                             linestyle="--", color="orange", label="Fitness máximo (no factible)")
                if "Fitness_max_factible" in df.columns:
                    plt.plot(df["Generacion"], df["Fitness_max_factible"],
                             linestyle="--", color="green", label="Fitness máximo (factible)")
            else:
                if "Fitness_max" in df.columns:
                    plt.plot(df["Generacion"], df["Fitness_max"],
                             linestyle="--", color="orange", label="Fitness máximo")

            # Línea vertical de convergencia
            if gen_conv is not None:
                plt.axvline(x=gen_conv, color="red", linestyle="--", linewidth=1, alpha=0.7)
            if gen_best is not None:
                plt.axvline(x=gen_best, color="blue", linestyle="--", linewidth=1, alpha=0.7)

            # Estética general
            plt.xlabel("Generación")
            plt.ylabel("Fitness (coste)")
            plt.legend(frameon=False)
            plt.grid(alpha=0.2)

            # Guardar figura
            salida = RUTA_SALIDA / f"{variante}_{archivo.stem}.png"
            plt.savefig(salida, bbox_inches="tight", dpi=300)
            plt.close()

        except Exception as e:
            print(f"⚠️ Error procesando {archivo.name}: {e}")

print("✅ Gráficas detalladas de convergencia generadas (sin etiquetas numéricas).")

# ============================================================
# GRÁFICAS COMPARATIVAS POR TAMAÑO DE POBLACIÓN
# ============================================================

print("📊 Generando gráficas comparativas por variante...")

for sub in RUTA_BASE.iterdir():
    if not sub.is_dir():
        continue

    variante = sub.name
    carpeta_gen = sub / "gen"
    if not carpeta_gen.exists():
        continue

    # Diccionario para asociar color/estilo a cada tamaño de población
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    plt.figure(figsize=(7, 4))

    for i, archivo in enumerate(sorted(carpeta_gen.glob("*.csv"))):
        try:
            df = pd.read_csv(archivo)
            df = df.groupby("Generacion", as_index=False).mean(numeric_only=True)

            # Extraer tamaño de población del nombre del archivo (Pxxxx)
            nombre = archivo.stem
            pop = "?"  # valor por defecto
            if "P" in nombre:
                try:
                    pop = int(nombre.split("P")[1].split("_")[0])
                except Exception:
                    pass

            plt.plot(df["Generacion"], df["Fitness_medio"],
                     label=f"Población {pop}",
                     color=colores[i % len(colores)])

        except Exception as e:
            print(f"⚠️ Error en comparativa ({archivo.name}): {e}")

    plt.xlabel("Generación")
    plt.ylabel("Fitness medio (coste)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)

    # Guardar comparativa
    salida_comp = RUTA_SALIDA / f"{variante}_comparativa_poblacion.png"
    plt.savefig(salida_comp, bbox_inches="tight", dpi=300)
    plt.close()
