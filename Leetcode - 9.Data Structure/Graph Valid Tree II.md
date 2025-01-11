
**Example 1**

```plain
输入:
addEdge(1, 2)
isValidTree()
addEdge(1, 3)
isValidTree()
addEdge(1, 5)
isValidTree()
addEdge(3, 5)
isValidTree()
输出: ["true","true","true","false"]
```



```python
class Solution:
    """
    @param a: the node a
    @param b: the node b
    @return: nothing
    """
    def __init__(self):
        self.edgeNum = 0
        self.vertexNum = 0
        self.tag = 1
        self.f = {}

    def Find(self, x):
        if self.f[x] == x:
            return x
        root = x
        while self.f[root] != root:
            root = self.f[root]
        while x != root:
            tmp = self.f[x]
            self.f[x] = root
            x = tmp
        return root

    def Union(self, x, y):
        fx = self.Find(x)
        fy = self.Find(y)
        if fx != fy:
            self.f[fx] = fy
        else:
            self.tag = 0

    def addEdge(self, a, b):
        if a not in self.f:
            self.f[a] = a
            self.vertexNum += 1
        if b not in self.f:
            self.f[b] = b
            self.vertexNum += 1
        self.edgeNum += 1
        self.Union(a, b)

        # write your code here

    """
    @return: check whether these edges make up a valid tree
    """

    def isValidTree(self):
        # write your code here
        if self.edgeNum + 1 == self.vertexNum and self.tag == 1:
            return True
        else:
            return False
```
pass