Lintcode 589
给一个图中的`n`个节点, 记为 `1` 到 `n` . 在开始的时候图中没有边。  
你需要完成下面两个方法:

1. `connect(a, b)`, 添加连接节点 `a`, `b` 的边.
2. `query(a, b)`, 检验两个节点是否联通

例1:
```python
"""
输入:
ConnectingGraph(5)
query(1, 2)
connect(1, 2)
query(1, 3) 
connect(2, 4)
query(1, 4) 
输出:
[false,false,true]
```
例2:
```python
"""
输入:
ConnectingGraph(6)
query(1, 2)
query(2, 3)
query(1, 3)
query(5, 6)
query(1, 4)

输出:
[false,false,false,false,false]
```



```python
class ConnectingGraph:
    """
    @param: n: An integer
    """
    def __init__(self, n):
        self.father = {}
        for i in range(1, n + 1):
            self.father[i] = i

    """
    @param: a: An integer
    @param: b: An integer
    @return: nothing
    """
    def connect(self, a, b):
        self.father[self.find(a)] = self.find(b)

    """
    @param: a: An integer
    @param: b: An integer
    @return: A boolean
    """
    def query(self, a, b):
        return self.find(a) == self.find(b)
        
    def find(self, node):
        path = []
        while self.father[node] != node:
            path.append(node)
            node = self.father[node]
            
        for n in path:
            self.father[n] = node
            
        return node
```
pass