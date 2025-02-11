Lintcode 3672
在本题中共存在 `n` 个城市，其编号范围为 1 到 n。

同时存在一个 `connections` 数组且 connections[i]=[ai,bi,ci]connections[i]=[ai​,bi​,ci​]，其含义为联通城市 aiai​ 和 bibi​ 的成本为 cici​。

请返回连通全部城市所需要的最低成本。如果无法联通全部城市，返回 **-1**。


**样例 1**
输入：
```python
"""
3
[[1,2,1], [2,3,2], [1,3,3]]
```
输出：
```python
"""
3
```
解释：
选择 `[1,2,1]` 和 `[2,3,2]` 即可联通全部 n 个城市，此时花费最少，为 3。

**样例 2**
输入：
```python
"""
3
[[1,2,1]]
```
输出：
```python
"""
-1
```
解释：
无法根据 `connections` 联通所有的城市。


```python
class Solution:
    """
    @param n: the number of cities
    @param connections: the connection info between cities
    @return: 
    """
    def __init__(self):
        self.p = []

    def minimum_cost(self, n: int, connections: List[List[int]]) -> int:
        # write your code here
        for i in range(n + 1):
            self.p.append(i)
        connections.sort(key=lambda x: x[2])
        ans = 0
        for a, b, cost in connections:
            if self.find(a) == self.find(b):
                continue
            self.p[self.find(a)] = self.find(b)
            ans += cost
            n -= 1
            if n == 1:
                return ans
        return -1

    
    def find(self, x: int) -> int:
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
```
pass