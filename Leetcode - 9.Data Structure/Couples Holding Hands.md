Lintcode 1043
`N`对夫妇坐在`2N`个排成一排的座位上. 现求最小的交换数量，使每对夫妇并坐一起，他们可以手牵着手。

一次交换可选择**任何**两个人交换座位。

人和座位由从`0`到`2N-1`的整数表示，夫妻按顺序编号，第一对是`(0,1)`，第二对是`(2,3)`，以此类推，最后一对是`(2N-2,2N-1)`。

初始座位由`row [i]`给出，表示坐在第`i`座位的人的编号。


**样例 1:**
```python
"""
输入: row = [0, 2, 1, 3]
输出: 1

解释: 只需交换row[1]的人和row[2]的人即可.
```
**样例 2:**
```python
"""
输入: row = [3, 2, 0, 1]
输出: 0

解释: 每一对夫妇已经并坐一起.
```


```python
from typing import (
    List,
)

class Solution:
    class UnionFind:
        def __init__(self, n):
            self.parent = [i for i in range(n)]
            self.count = n

        def getCount(self):
            return self.count

        def find(self, x):
            while x != self.parent[x]:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def uni(self, x, y):
          rootX = self.find(x)
          rootY = self.find(y)
          if rootX == rootY:
              return
          self.parent[rootX] = rootY
          self.count -= 1

    def min_swaps_couples(self, row: List[int]) -> int:
        length = len(row)
        N = length // 2
        unionFind = self.UnionFind(N)
        for i in range(0, length, 2):
            unionFind.uni(row[i] // 2, row[i + 1] // 2)
        return N - unionFind.getCount()
```
pass