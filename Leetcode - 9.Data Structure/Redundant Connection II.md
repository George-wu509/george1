
样例 1:
```.
输入: [[1,2], [1,3], [2,3]]
输出: [2,3]
解释: 给定的有向图将是这样的:
  1
 / \
v   v
2-->3
```
样例 2:
```.
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