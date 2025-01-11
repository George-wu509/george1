
样例 1：
```
输入：
logs = [[20220101,0,1],[20220109,3,4],[20220304,0,4],[20220305,0,3],[20220404,2,4]]
n = 5
输出：
20220404
解释：
time = 20220101，0 和 1 成为好友，关系为：[0,1], [2], [3], [4]
time = 20220109，3 和 4 成为好友，关系为：[0,1], [2], [3,4]
time = 20220304，0 和 4 成为好友，关系为：[0,1,3,4], [2]
time = 20220305，0 和 3 已经是好友
time = 20220404，2 和 4 成为好友，关系为：[0,1,2,3,4]，所有人都相互认识
```
样例 2：
```
输入：
logs = [[7,3,1],[3,0,3],[2,0,1],[1,1,2],[5,3,2]]
n = 4
输出：
3
```




```python
from typing import (
    List,
)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            if self.size[pa] > self.size[pb]:
                self.parent[pb] = pa
                self.size[pa] += self.size[pb]
            else:
                self.parent[pa] = pb
                self.size[pb] += self.size[pa]

class Solution:
    """
    @param logs: Logging with time, x, y
    @param n: Total count of people
    @return: Timestamp of when everyone became friends
    """
    def earliest_acq(self, logs: List[List[int]], n: int) -> int:
        uf = UnionFind(n)
        for time, x, y in sorted(logs):
            if uf.find(x) == uf.find(y):
                continue
            uf.union(x, y)
            n -= 1
            if n == 1:
                return time
        return -1
```
pass