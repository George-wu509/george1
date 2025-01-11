
**样例 1:**
```
输入； x = [1,1], y = [2,3], a =[1,2], b = [2,3]
输出：[2,1]
解释：1与2是父子关系，2与3是兄弟关系，它们的共同父节点为1。
```
**样例 2：**
```
输入：x = [1,1,2], y =[2,3,4], a = [1,2,1], b = [2,3,4]
输出：[2,1,0]。
解释：1与2是父子关系，2与3是兄弟关系，它们的共同父节点为1，1与4不是兄弟关系也不是父子关系。
```


```python
class Solution:
    """
    @param x: The x
    @param y: The y
    @param a: The a
    @param b: The b
    @return: The Answer
    """
    def tree(self, x, y, a, b):
        graph = self.build_graph(x, y)
        parent = self.build_tree(graph)
        
        results = []
        for u, v in zip(a, b):
            if parent[u] == parent[v]:
                results.append(1)
            elif parent[u] == v or parent[v] == u:
                results.append(2)
            else:
                results.append(0)
        return results
    
    def build_graph(self, x, y):
        graph = {}
        for u, v in zip(x, y):
            if u not in graph:
                graph[u] = set()
            if v not in graph:
                graph[v] = set()
            graph[u].add(v)
            graph[v].add(u)
        return graph
        
    def build_tree(self, graph):
        from collections import deque
        visited = set([1])
        queue = deque([1])
        parent = {1: None}
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = node
        return parent
```
pass