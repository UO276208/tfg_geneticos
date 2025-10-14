import pandas as pd
import numpy as np
from pathlib import Path

class DataLogger:
    def __init__(self,n_gens):
        self.fitnesses_max = np.zeros(n_gens, dtype=int)
        self.i_f_max = 0

        self.fitnesses_avg = np.zeros(n_gens, dtype=int)
        self.i_f_avg = 0

        self.n_violaciones = np.zeros(n_gens, dtype=int)
        self.i_n_violaciones = 0

        self.fitnesses_max_factibles = np.zeros(n_gens, dtype=int)
        self.i_ff = 0

        #self.fitnesses_max_no_factibles = np.zeros(n_gens, dtype=int)
        #self.i_fnf = 0

        self.time = 0

    def add_fitness(self, value_max, value_avg, value_n_violaciones, value_max_factible = None):
        self.fitnesses_max[self.i_f_max] = value_max
        self.i_f_max += 1
        self.fitnesses_avg[self.i_f_avg] = value_avg
        self.i_f_avg += 1
        self.n_violaciones[self.i_n_violaciones] = value_n_violaciones
        self.i_n_violaciones += 1
        if value_max_factible != None:
            self.fitnesses_max_factibles[self.i_ff] = value_max_factible
            self.i_ff +=1

    #def add_fitness_f(self, value):
        #self.fitnesses_max_factibles[self.i_ff] = value
        #self.i_ff += 1

    #def add_fitness_nf(self, value):
        #self.fitnesses_max_no_factibles[self.i_fnf] = value
        #self.i_fnf += 1

    def set_time(self, time):
        self.time = time

    def write_file(self, penalizaciones, path, data_max, data_medio, data_n_violaciones, data_max_factible, name):
        path.mkdir(parents=True, exist_ok=True)
        if penalizaciones:
            df = pd.DataFrame({
                'Generacion': range(1, len(data_max) + 1),
                'Fitness_max': data_max,
                'Fitness_medio': data_medio,
                'Fitness_max_factible': data_max_factible,
                'N_violaciones': data_n_violaciones,
                'Tiempo': self.time  # Se asigna el mismo valor a cada fila
            })
            df.to_csv(Path(path, name), index=False)
        else:
            df = pd.DataFrame({
                'Generacion': range(1, len(data_max) + 1),
                'Fitness_max': data_max,
                'Fitness_medio': data_medio,
                'N_violaciones' : data_n_violaciones,
                'Tiempo': self.time  # Se asigna el mismo valor a cada fila
            })
            df.to_csv(Path(path, name), index=False)


    def write(self, nombre, directorio_nombre):
        ruta = Path(Path(__file__).resolve().parent.parent, 'SSGAs', 'data2', directorio_nombre)
        if self.i_ff < 0 < self.i_f_max and self.i_f_avg > 0:
            self.write_file(False, ruta, self.fitnesses_max, self.fitnesses_avg, self.n_violaciones, nombre+ '.csv')
        elif self.i_f_max > 0 and self.i_f_avg > 0:
            self.write_file(True, ruta, self.fitnesses_max, self.fitnesses_avg, self.n_violaciones, self.fitnesses_max_factibles, nombre + '.csv')

        #if self.i_ff > 0:
            #self.write_file(ruta, self.fitnesses_max, nombre+ '_ff.csv')
        #if self.i_fnf > 0:
        # self.write_file(ruta, self.fitnesses_max, nombre + '_fnf.csv')