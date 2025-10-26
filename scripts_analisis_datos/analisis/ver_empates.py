import pandas as pd
import numpy as np
from pathlib import Path
import re

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_BASE = Path(__file__).resolve().parent.parent.parent / "SSGAs" / "data_copia"
TOLERANCIA_GEN = 10  # generaciones de diferencia permitidas en el empate

# ------------------------------------------------------------
# Utilidad: extraer parámetros del nombre de archivo (si procede)
# Formato esperado tipo: ...P{pop}_G{gen}_{mut}-gamma_{gamma}.csv
# ------------------------------------------------------------
PAT = re.compile(r"P(\d+)_G(\d+)_(\d+(?:\.\d+)?)-gamma_(\d+(?:\.\d+)?)")

def extraer_parametros(nombre):
    m = PAT.search(nombre)
    if not m:
        return (None, None, None, None)
    pop, gen, mut, gamma = m.groups()
    return int(pop), int(gen), float(mut), float(gamma)

# ============================================================
# RECOGER TODAS LAS CONFIGURACIONES (SOLO medias/, EXCEPTO gen)
# ============================================================

registros = []

for carpeta_variante in RUTA_BASE.iterdir():
    if not carpeta_variante.is_dir():
        continue

    variante = carpeta_variante.name

    # subcarpetas: gamma / mut / pop / gen
    for carpeta_exp in carpeta_variante.iterdir():
        if not carpeta_exp.is_dir():
            continue
        if carpeta_exp.name == "gen":
            continue  # ignorar convergencia

        carpeta_medias = carpeta_exp / "medias"
        if not carpeta_medias.exists():
            # Si no hay 'medias', saltamos (evita leer CSV crudos)
            continue

        for csv_path in carpeta_medias.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)

                # Por si acaso, aplicar media por generación (aunque en 'medias' ya lo esté)
                df = df.groupby("Generacion", as_index=False).mean(numeric_only=True)

                # Selección del fitness correcto (coste → cuanto más bajo, mejor)
                if "penalizaciones" in variante and "Fitness_max_factible" in df.columns:
                    # Limpiar sentinelas -1 (no factibles)
                    fact = df["Fitness_max_factible"].replace(-1, np.nan)
                    # Si no hay factibles en toda la ejecución, NO cuenta para empate
                    if fact.dropna().empty:
                        continue
                    mejor_fit = fact.min()
                    # generación mínima donde aparece ese mejor factible
                    gen_mejor = df.loc[fact == mejor_fit, "Generacion"].min()
                else:
                    # No penalizadas: mejor fitness general
                    if "Fitness_max" not in df.columns:
                        continue
                    mejor_fit = df["Fitness_max"].min()
                    gen_mejor = df.loc[df["Fitness_max"] == mejor_fit, "Generacion"].min()

                # Tiempo total (última fila) y tiempo estimado al mejor (proporcionalidad)
                tiempo_total = df["Tiempo"].iloc[-1] if "Tiempo" in df.columns else np.nan
                gen_final = df["Generacion"].max()
                if pd.notna(tiempo_total) and gen_final > 0:
                    tiempo_mejor = (tiempo_total * gen_mejor) / gen_final
                else:
                    tiempo_mejor = np.nan

                pop, gen_cfg, mut, gamma = extraer_parametros(csv_path.name)

                registros.append({
                    "Variante": variante,
                    "Archivo": csv_path.name,
                    "Experimento": carpeta_exp.name,  # gamma / mut / pop
                    "Generacion_mejor": gen_mejor,
                    "Fitness_mejor": mejor_fit,
                    "Tiempo_total": tiempo_total,
                    "Tiempo_mejor": tiempo_mejor,
                    "Tiempo_redondeado": round(tiempo_total) if pd.notna(tiempo_total) else np.nan,
                    "Pop": pop,
                    "Gen_cfg": gen_cfg,
                    "Mut": mut,
                    "Gamma": gamma,
                })

            except Exception as e:
                print(f"⚠️ Error procesando {csv_path}: {e}")

df = pd.DataFrame(registros)

# Si no hay datos, salir limpio
if df.empty:
    print("No se encontraron datos en 'medias/'. ¿Ruta correcta?")
    raise SystemExit

# ============================================================
# DETECTAR EMPATES EN PRIMERA POSICIÓN
# ============================================================

print("\n🔎 Buscando empates en la mejor configuración (fitness mínimo, tiempo redondeado igual y generación cercana ±10)...\n")

for variante, grupo in df.groupby("Variante"):
    # Mejor fitness (mínimo coste) de la variante
    mejor_fit = grupo["Fitness_mejor"].min()
    top = grupo[grupo["Fitness_mejor"] == mejor_fit].copy()

    if top.empty:
        print(f"{variante}: sin mejores válidos.")
        continue

    # Agrupar por Tiempo_redondeado para detectar empates reales con mismo tiempo total (redondeado)
    hubo_empates = False
    for tiempo_r, gtime in top.groupby("Tiempo_redondeado"):
        if len(gtime) < 2:
            continue

        # Ordenar por generación para emparejar vecinos cercanos
        gtime = gtime.sort_values("Generacion_mejor")

        # Comprobar diferencias de generación dentro de la tolerancia
        candidatos = []
        filas = list(gtime.to_dict("records"))
        for i in range(len(filas)):
            for j in range(i+1, len(filas)):
                if abs(filas[i]["Generacion_mejor"] - filas[j]["Generacion_mejor"]) <= TOLERANCIA_GEN:
                    candidatos.append((filas[i], filas[j]))

        if candidatos:
            hubo_empates = True
            print(f"⚠️ {variante}: empates con fitness={mejor_fit:.6f} y tiempo≈{int(tiempo_r)}s")
            for a, b in candidatos:
                print(f"  • {a['Archivo']}  ↔  {b['Archivo']}")
                print(f"    Gen: {a['Generacion_mejor']} vs {b['Generacion_mejor']} | "
                      f"Tiempo: {a['Tiempo_total']:.2f}s vs {b['Tiempo_total']:.2f}s")
                # (Opcional) mostrar parámetros para entender qué cambia
                print(f"    Params A: pop={a['Pop']} gen={a['Gen_cfg']} mut={a['Mut']} gamma={a['Gamma']}")
                print(f"    Params B: pop={b['Pop']} gen={b['Gen_cfg']} mut={b['Mut']} gamma={b['Gamma']}\n")

    if not hubo_empates:
        # Si no hay empates, reportar la mejor configuración única (o múltiples con distinto tiempo/generación)
        mejor_filas = top.sort_values(["Tiempo_redondeado", "Generacion_mejor"]).head(3)
        print(f"✅ {variante}: sin empates estrictos. Mejores candidatos:")
        print(mejor_filas[["Archivo", "Fitness_mejor", "Tiempo_total", "Generacion_mejor"]].to_string(index=False))
        print()

print("\n✅ Análisis completado.")
