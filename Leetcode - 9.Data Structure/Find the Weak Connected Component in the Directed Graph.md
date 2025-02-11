Lintcode 432
请找出有向图中弱连通分量。图中的每个节点包含 1 个标签和1 个相邻节点列表。（有向图的弱连通分量是任意两点均有有向边相连的极大子图）

**样例 1:**
```python
"""
输入: {1,2,4#2,4#3,5#4#5#6,5}
输出: [[1,2,4],[3,5,6]]
解释: 
  1----->2    3-->5
   \     |        ^
    \    |        |
     \   |        6
      \  v
       ->4
```
**样例 2:**
```python
"""
输入: {1,2#2,3#3,1}
输出: [[1,2,3]]
```


```python
class Solution:
    def __init__(self):
        self.f = {}

    def merge(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.f[x] = y

    def find(self, x):
        if self.f[x] == x:
            return x
        
        self.f[x] = self.find(self.f[x])
        return self.f[x]

    # @param {DirectedGraphNode[]} nodes a array of directed graph node
    # @return {int[][]} a connected set of a directed graph
    def connectedSet2(self, nodes):
        for node in nodes:
            self.f[node.label] = node.label

        for node in nodes:
            for nei in node.neighbors:
                self.merge(node.label, nei.label)

        result = []
        g = {}
        cnt = 0

        for node in nodes:
            x = self.find(node.label)
            if not x in g:
                cnt += 1
                g[x] = cnt
            
            if len(result) < cnt:
                result.append([])
        
            result[g[x] - 1].append(node.label)

        return result
```
pass