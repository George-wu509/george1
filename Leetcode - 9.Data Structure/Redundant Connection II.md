Lintcode 1087
在这个问题中，有根树是一种这样的有向图，它只有一个根节点，所有其他节点都是该节点的后代，加上每个节点只有一个父母节点，（除了根节点没有父母）。

给定的输入是一个有向图，它以具有N个节点（具有不同的值1,2，...，N）的有根树开始，并添加了一个额外的有向边。 添加的边有两个不同的顶点，从1到N中选择，并且不是已经存在的边。

得到的图形以边的2D数组给出。 边的每个元素是一对[u，v]，表示连接节点u和v的有向边，其中u是子v的父节点。

返回可以删除的边，以便生成的图是N个节点的有根树。 如果有多个答案，则返回给定2D数组中最后出现的答案。


样例 1:
```python
"""
输入: [[1,2], [1,3], [2,3]]
输出: [2,3]
解释: 给定的有向图将是这样的:
  1
 / \
v   v
2-->3
```
样例 2:
```python
"""
输入: [[1,2], [2,3], [3,4], [4,1], [1,5]]
输出: [4,1]
解释: 给定的有向图将是这样的:
5 <- 1 -> 2
     ^    |
     |    v
     4 <- 3
```


```python
from typing import (
    List,
)
class UnionFind:
    def __init__(self, n):
        self.ancestor = list(range(n))
    
    def union(self, index1: int, index2: int):
        self.ancestor[self.find(index1)] = self.find(index2)
    
    def find(self, index: int) -> int:
        if self.ancestor[index] != index:
            self.ancestor[index] = self.find(self.ancestor[index])
        return self.ancestor[index]

class Solution:
    """
    @param edges: List[List[int]]
    @return: return List[int]
    """
    def find_redundant_directed_connection(self, edges: List[List[int]]) -> List[int]:
        # write your code here
        n = len(edges)
        uf = UnionFind(n + 1)
        parent = list(range(n + 1))
        conflict = -1
        cycle = -1
        for i, (node1, node2) in enumerate(edges):
            if parent[node2] != node2:
                conflict = i
            else:
                parent[node2] = node1
                if uf.find(node1) == uf.find(node2):
                    cycle = i
                else:
                    uf.union(node1, node2)

        if conflict < 0:
            return [edges[cycle][0], edges[cycle][1]]
        else:
            conflictEdge = edges[conflict]
            if cycle >= 0:
                return [parent[conflictEdge[1]], conflictEdge[1]]
            else:
                return [conflictEdge[0], conflictEdge[1]]
```
pass