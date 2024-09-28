import numpy as np
import matplotlib.pyplot as plt

def get_number_distribution(i,j):
    gamma = 1.2
    n = np.random.normal(0,1)
    return round((1+gamma)**n, 3)

array = np.fromfunction(np.vectorize(get_number_distribution), (1, 1000000), dtype=float)

unique, counts = np.unique(array, return_counts=True)

x = np.linspace(-1,1.5)                # definimos un vector con 50 elementos en (-1,1.5)
ox = 0*x

plt.figure()
plt.plot(unique, counts)

plt.show()