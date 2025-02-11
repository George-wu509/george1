Lintcode 3670
在一个社交圈子中，有 `n` 个人，编号从 `0` 到 `n - 1`。现在有一份日志列表 `logs`，其中 `logs[i] = [time, x, y]` 表示 `x` 和 `y` 在 `time` 时刻成为朋友相互认识。

友谊是 **相互且具有传递性** 的。 也就是说：

- 相互性：当 `a` 和 `b` 成为朋友，那么 `b` 的朋友中也有 `a`
- 传递性：当 `a` 和 `b` 成为朋友，`b` 的朋友中有 `c`，那么 `a` 和 `c` 也会成为朋友相互认识

返回这个圈子中所有人之间都相互认识的最早时间。如果找不到最早时间，则返回 `-1`


样例 1：
```python
"""
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
```python
"""
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