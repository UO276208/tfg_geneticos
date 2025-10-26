import os
import re

# Nombre del archivo (sin caracteres inválidos)
def extraer_info_algoritmo(nombre_archivo):
    if nombre_archivo.startswith("prim"):
        algoritmo = "prim"
    elif nombre_archivo.startswith("kruskal"):
        algoritmo = "kruskal"
    else:
        algoritmo = "desconocido"

    penalizaciones = "con_penalizaciones" if "penalizaciones" in nombre_archivo else "sin_penalizaciones"
    return algoritmo, penalizaciones

def extraer_parametros(nombre_archivo):
    """
    Extrae pop, gen, mut y gamma de un nombre de archivo con formato:
    ...P{pop}_G{gen}_{mut}-gamma_{gamma}...
    """
    pat = re.compile(r"P(\d+)_G(\d+)_(\d+\.\d+)-gamma_(\d+\.\d+)")
    m = pat.search(nombre_archivo)
    if not m:
        return None, None, None, None
    pop, gen, mut, gamma = m.groups()
    return int(pop), int(gen), float(mut), float(gamma)

#algoritmo, penalizaciones = extraer_info_algoritmo('kruskal_imp_h_bayg29_muttestP2000_G20000_0.06-gamma_0.3.csv')
#print(algoritmo)
#print(penalizaciones)

pop, gen, mut, gamma = extraer_parametros('kruskal_imp_h_bayg29_muttestP2000_G20000_0.06-gamma_0.3.csv')
print(pop)
print(gen)
print(mut)
print(gamma)
