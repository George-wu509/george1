Lintcode 178
给出 `n` 个节点，标号分别从 `0` 到 `n - 1` 并且给出一个 `无向` 边的列表 (给出每条边的两个顶点), 写一个函数去判断这张｀无向｀图是否是一棵树


**样例 1:**
```python
"""
输入: n = 5 edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
输出: true.
```
**样例 2:**
```python
"""
输入: n = 5 edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]
输出: false.
```



```python
class Solution:
    """
    @param n: An integer
    @param edges: a list of undirected edges
    @return: true if it's a valid tree, or false
    """
    def validTree(self, n, edges):
        if n - 1 != len(edges):
            return False
        
        self.father = {i: i for i in range(n)}
        self.size = n
        
        for a, b in edges:
            self.union(a, b)
            
        return self.size == 1
        
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.size -= 1
            self.father[root_a] = root_b
        
    def find(self, node):
        path = []
        while node != self.father[node]:
            path.append(node)
            node = self.father[node]
            
        for n in path:
            self.father[n] = node
            
        return node
```
pass