from fitness import prim_ADAP as prim
import numpy as np

grafo = [[0,4,10,3,2],
         [4, 0, 1, 5, 1],
         [10,1, 0, 2, 4],
         [3, 5, 2, 0 ,6],
         [2, 1, 4, 6, 0]]

MST_completar = [[2,0,4],[1,4,1]]

grafo_prim = prim.Graph_prim(grafo, 4, MST_completar)

MST = grafo_prim.prim([0.381, 0.88, 1.16, 1.057, 1.034])
print(MST)