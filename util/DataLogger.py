import pandas as pd
import numpy as np
from pathlib import Path

class DataLogger:
    def __init__(self,n_gens):
        self.fitnesses = np.zeros(n_gens, dtype=int)
        self.i_f = 0
        self.fitnesses_factibles = np.zeros(n_gens, dtype=int)
        self.i_ff = 0
        self.fitnesses_no_factibles = np.zeros(n_gens, dtype=int)
        self.i_fnf = 0
        self.time = 0

    def add_fitness(self, value):
        self.fitnesses[self.i_f] = value
        self.i_f += 1

    def add_fitness_f(self, value):
        self.fitnesses_factibles[self.i_ff] = value
        self.i_ff += 1

    def add_fitness_nf(self, value):
        self.fitnesses_no_factibles[self.i_fnf] = value
        self.i_fnf += 1

    def set_time(self, time):
        self.time = time

    def test(self):
        print(Path(Path(__file__).resolve().parent.parent, 'SSGAs', 'data2'))

    def write_file(self, path, data, name):
        path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            'Generacion': range(1, len(data) + 1),
            'Fitness': data,
            'Tiempo': self.time  # Se asigna el mismo valor a cada fila
        })
        df.to_csv(Path(path, name), index=False)


    def write(self, nombre, directorio_nombre):
        ruta = Path(Path(__file__).resolve().parent.parent, 'SSGAs', 'data2', directorio_nombre)
        if self.i_f > 0:
            self.write_file(ruta, self.fitnesses, nombre+ '_f.csv')
        if self.i_ff > 0:
            self.write_file(ruta, self.fitnesses, nombre+ '_ff.csv')
        if self.i_fnf > 0:
            self.write_file(ruta, self.fitnesses, nombre + '_fnf.csv')