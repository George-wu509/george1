
**样例 1:**
```
输入: {1,2,4#2,1,4#3,5#4,1,2#5,3}
输出: [[1,2,4],[3,5]]
解释: 

  1------2  3
   \     |  | 
    \    |  |
     \   |  |
      \  |  |
        4   5
```
**样例 2:**
```
输入: {1,2#2,1}
输出: [[1,2]]
解释:

  1--2
```



```python
class Solution:
    # @param {UndirectedGraphNode[]} nodes a array of undirected graph node
    # @return {int[][]} a connected set of a undirected graph
    def dfs(self, x, tmp):
        self.v[x.label] = True
        tmp.append(x.label)
        for node in x.neighbors:
            if not self.v[node.label]:
                self.dfs(node, tmp)
            
    def connectedSet(self, nodes):
        # Write your code here
        self.v = {}
        for node in nodes:
            self.v[node.label] = False

        ret = []
        for node in nodes:
            if not self.v[node.label]:
                tmp = []
                self.dfs(node, tmp)
                ret.append(sorted(tmp))
        return ret
```
pass