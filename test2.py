def check_degree_limit(edge, MST):
    degree_limit = 2
    degree = 0
    for mst_edge in MST:
        if mst_edge[1] == edge[1] or mst_edge[2] == edge[1]:
            degree += 1
        if degree > (degree_limit - 1):
            return False
    return True


def check_degree_limit2(edge, MST):
    degree_limit = 2
    # Inicializar un diccionario para contar los grados de los nodos
    node_degrees = {}

    # Calcular los grados actuales de los nodos en el MST
    for mst_edge in MST:
        node_degrees[mst_edge[1]] = node_degrees.get(mst_edge[1], 0) + 1
        node_degrees[mst_edge[2]] = node_degrees.get(mst_edge[2], 0) + 1

    # Verificar grados tras añadir el nuevo arco
    for node in edge[1:]:  # Verifica ambos nodos del arco
        # Incrementar temporalmente el grado
        current_degree = node_degrees.get(node, 0) + 1
        # Comparar con el límite de grado
        if current_degree > degree_limit:
            return False

    return True


MST = [[23, 1, 2], [21, 2, 3]]
edge = [2, 4, 2]
print('Primera: ' + str(check_degree_limit(edge, MST)))
print('Segunda: ' + str(check_degree_limit2(edge, MST)))
