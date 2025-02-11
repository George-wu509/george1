Lintcode 591
给一个图中的 `n` 个节点，记为 `1` 到 `n` 。`ConnectingGraph3(n)` 会创建 `n` 个节点，在开始的时候图中没有边。  
你需要完成下面两个方法：

1. `connect(a, b)`，添加一条连接节点 a, b的边
2. `query()`，返回图中连通区域个数


例1:
```python
"""
输入:
ConnectingGraph3(5)
query()
connect(1, 2)
query()
connect(2, 4)
query()
connect(1, 4)
query()

输出:[5,4,3,3]
```
例2:
```python
"""
输入:
ConnectingGraph3(6)
query()
query()
query()
query()
query()


输出:
[6,6,6,6,6]
```


```python
class ConnectingGraph3:
    """
    @param: n: An integer
    """
    def __init__(self, n):
        self.size = n
        self.father = {}
        for i in range(1, n + 1):
            self.father[i] = i

    """
    @param: a: An integer
    @param: b: An integer
    @return: nothing
    """
    def connect(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.father[root_a] = root_b
            self.size -= 1
            
    """
    @return: An integer
    """
    def query(self):
        return self.size

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