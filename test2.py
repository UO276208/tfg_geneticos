from functools import reduce

x = [1,2,3,4]
y = reduce(lambda a, b: str(a) + '\n' + str(b), x)

print(y)