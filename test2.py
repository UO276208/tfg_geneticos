import numpy as np

s = [(4.155456, 0, 1), (3.6117360000000005, 0, 2), (6.93036, 0, 3), (4.155456, 0, 1), (32.837136, 1, 2), (26.2539, 1, 3), (3.6117360000000005, 0, 2), (32.837136, 1, 2), (3.042495, 2, 3), (6.93036, 0, 3), (26.253899999999998, 1, 3), (3.042495, 2, 3)]
s.sort(key=lambda x: x[0])
print(s)