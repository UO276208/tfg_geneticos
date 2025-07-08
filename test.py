import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


# Función generadora de valores (solo depende de gamma)
def generar_valor(i, j, gamma):
    # Ejemplo: distribución exponencial con media 1/gamma
    return np.random.exponential(scale=1 / gamma)


# Lista de valores de gamma a probar
gammas = [0.5, 1.2, 2.0, 3.0]

# Número de muestras por gamma
n_muestras = 10000

# Parámetros fijos i y j
i, j = 0, 0

# Crear figura
plt.figure(figsize=(10, 6))

# Dibujar cada curva
for gamma in gammas:
    muestras = [generar_valor(i, j, gamma) for _ in range(n_muestras)]

    kde = gaussian_kde(muestras)
    x_vals = np.linspace(min(muestras), max(muestras), 500)
    y_vals = kde(x_vals)

    plt.plot(x_vals, y_vals, label=f'gamma = {gamma}')  # Cada línea con etiqueta

# Configuración final
plt.title('Curvas de densidad estimada para distintos valores de gamma')
plt.xlabel('Valor generado')
plt.ylabel('Densidad estimada')
plt.legend(title="Valores de gamma")  # Leyenda con título opcional
plt.grid(True)
plt.tight_layout()
plt.show()
