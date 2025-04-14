import os
from functools import reduce


class ResultsWriter:
    def __init__(self):
        self.fitnesses = []
        self.fitnesses_factibles = []
        self.fitnesses_no_factibles = []
        self.time = 0

    def add_fitness(self, value):
        self.fitnesses.append(value)

    def add_fitness_f(self, value):
        self.fitnesses_factibles.append(value)

    def add_fitness_nf(self, value):
        self.fitnesses_no_factibles.append(value)

    def set_time(self, time):
        self.time = time

    def write_file(self, list, nombre, directorio_nombre):
        if len(list) > 0:
            text = reduce(lambda a, b: str(a) + '\n' + str(b), list)
            text += '\n' + str(self.time)
            ruta = 'C:/Users/Ines/PycharmProjects/tfg_geneticos/SSGAs/data2/'+directorio_nombre+'/'
            os.makedirs(ruta, exist_ok=True)

            with open(ruta+nombre, 'w') as f:
                f.write(text)
            list.clear()

    def write(self, nombre, directorio_nombre):
        self.write_file(self.fitnesses, nombre + '_f.txt', directorio_nombre)
        self.write_file(self.fitnesses_factibles, nombre + '_ff.txt', directorio_nombre)
        self.write_file(self.fitnesses_no_factibles, nombre + '_fnf,txt', directorio_nombre)
