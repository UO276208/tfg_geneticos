import os


def read_LOWER_DIAG_ROW(size, numbers):
    size = 26
    matriz = [[0 for _ in range(size)] for _ in range(size)]
    indice = 0
    for i in range(size):
        for j in range(i + 1):  # Llenar la diagonal y debajo de ella
            matriz[i][j] = numbers[indice]
            matriz[j][i] = numbers[indice]  # Reflejar en la parte superior
            indice += 1
    return matriz


def read_matrix(name):
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    ruta_archivo = os.path.join(ruta_actual, "..", name)
    with  open(ruta_archivo, mode="r") as archivo:
        empieza_matriz = False
        size = 0
        numbers = []
        format = ''

        for linea in archivo:
            if linea[0:5] == '1TYPE' and linea[7:] != 'TSP\n' or linea == 'EOF\n':
                break
            if linea[0:9] == 'DIMENSION':
                size = int(linea[11:(len(linea) - 1)])
            if linea[0:18] == 'EDGE_WEIGHT_FORMAT' and linea[20:(len(linea) - 2)] == 'LOWER_DIAG_ROW':
                format = linea[20:(len(linea) - 2)]
            if linea == "EDGE_WEIGHT_SECTION\n":
                empieza_matriz = True
            if empieza_matriz and linea != "EDGE_WEIGHT_SECTION\n":
                numbers.append(int(linea[0:(len(linea) - 1)]))

        if format == 'LOWER_DIAG_ROW':
            return read_LOWER_DIAG_ROW(size, numbers)

#matrizz = read_matrix("fri26.tsp")