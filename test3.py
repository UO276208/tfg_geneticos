class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u == root_v:
            return False

        if self.rank[root_u] > self.rank[root_v]:
            self.parent[root_v] = root_u
        elif self.rank[root_u] < self.rank[root_v]:
            self.parent[root_u] = root_v
        else:
            self.parent[root_v] = root_u
            self.rank[root_u] += 1

        return True

MST_completar = [[2,0,4],[1,4,1], [3,4,2], [3,2,1]]
uf = UnionFind(5)

for c, u, v in MST_completar:
    if not uf.union(u,v):
        print('Tiene ciclos')
        break
