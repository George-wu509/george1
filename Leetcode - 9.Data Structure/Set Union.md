Lintcode 1396
有一个集合组成的list，如果有两个集合有相同的元素，将他们合并。返回最后还剩下几个集合。

**样例1:**
```python
"""
输入：list = [[1,2,3],[3,9,7],[4,5,10]]
输出：2 .
样例：剩下[1,2,3,9,7]和[4,5,10]这2个集合。
```
**样例 2:**
```python
"""
输入：list = [[1],[1,2,3],[4],[8,7,4,5]]
输出 ：2
解释：剩下[1,2,3]和[4,5,7,8] 2个集合。
```



```python
class UnionFind:
    def __init__(self):
        self.leaders = {}
        self.ranks = {}
        self.cnt = 0
    
    def add(self, x):
        if x in self.leaders:
            return
        self.leaders[x] = x
        self.ranks[x] = 1
        self.cnt += 1
    
    def find(self, x):
        if self.leaders[x] != x:
            self.leaders[x] = self.find(self.leaders[x])
        return self.leaders[x]

    def union(self, x, y):
        p = self.find(x)
        q = self.find(y)
        if p == q: 
            return False
        self.cnt -= 1
        if self.ranks[p] < self.ranks[q]:
            self.leaders[p] = q
        elif self.ranks[p] > self.ranks[q]:
            self.leaders[q] = p
        else:        
            self.leaders[q] = p
            self.ranks[p] += 1
        return True
        
class Solution:
    def set_union(self, sets):
        uf = UnionFind()
        for grp in sets:
            for i in range(len(grp)):
                uf.add(grp[i])
                uf.union(grp[i], grp[0])
        return uf.cnt
```
pass