
例1:
```
输入:
ConnectingGraph2(5)
query(1)
connect(1, 2)
query(1)
connect(2, 4)
query(1)
connect(1, 4)
query(1)

输出:
[1,2,3,3]

```
例2:
```
输入:
ConnectingGraph2(6)
query(1)
query(2)
query(1)
query(5)
query(1)


输出:
[1,1,1,1,1]
```


```python
class ConnectingGraph2:
    """
    @param: n: An integer
    """
    def __init__(self, n):
        self.father = {}
        self.count = {}
        for i in range(1, n + 1):
            self.father[i] = i
            self.count[i] = 1

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
            self.count[root_b] += self.count[root_a]

    """
    @param: a: An integer
    @return: An integer
    """
    def query(self, a):
        return self.count[self.find(a)]

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