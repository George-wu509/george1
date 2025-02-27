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


### 解题思路

本题主要考查的知识点为 Kruskal 算法

Kruskal 算法：从一个无向有权图中找到最小生成树，其核心思路为贪心的将所有的边按照权值进行排序，然后使用并查集维护图的连通分量，通过不断从小到大遍历所有的边来连通还未连通的端点，当所有的端点都被联通时，我们则找到了最小生成树。

本题可根据 Kruskal 算法的同样的思路，来计算连通所有城市的成本。首先 将 `connections` 数组进行排序。然后开始遍历 `connections`。

在遍历的过程中，我们需要通过并查集来查看两个端点的连通状态。因此我们可以初始化并查集的 `p` （parent） 数组。在遍历的过程中，如果两个节点不属于同一个连通分量，我们则将其连接，同时增加 `ans`（连通总成本）。在每个连通之后，我们可知独立的端点将减少 1 个，因此我们可以判断当独立端点已经只剩一个时，则代表所有的端点都完成了连通，此时我们可直接返回 `ans`。

如果遍历完全部边后独立的端点仍然没用减为 1，则表示我们无法连通全部城市，此时返回 -1 即可。

解釋:
![[Pasted image 20250226221217.png]]