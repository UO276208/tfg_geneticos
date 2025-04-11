import pandas as pd
import numpy as np

class DataLogger:
    def __init__(self,n_gens):
        self.fitnesses = np.zeros(n_gens, dtype=int)
        self.fitnesses_factibles = np.zeros(n_gens, dtype=int)
        self.fitnesses_no_factibles = np.zeros(n_gens, dtype=int)
        self.time = 0

    def add_fitness(self, value):
        self.fitnesses.append(value)

    def add_fitness_f(self, value):
        self.fitnesses_factibles.append(value)

    def add_fitness_nf(self, value):
        self.fitnesses_no_factibles.append(value)

    def set_time(self, time):
        self.time = time

    def write_file(self, list, nombre):
        if len(list) > 0:
            print()

    def write(self, nombre):
        self.write_file(self.fitnesses, nombre + '_f.txt')
        self.write_file(self.fitnesses_factibles, nombre + '_ff.txt')
        self.write_file(self.fitnesses_no_factibles, nombre + '_fnf,txt')
