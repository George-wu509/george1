
LintCode 605 **Sequence Reconstruction** 要求验证一个唯一的排列 `org` 是否可以由给定的 `seqs` 重建。具体而言，使用 **拓扑排序（BFS）** 来判断序列是否唯一且合法。

---

### 问题描述

判断是否序列 `org` 能唯一地由 `seqs`重构得出. `org`是一个由从1到n的正整数**排列**而成的序列，1≤n≤10^4。 重构表示组合成`seqs`的一个最短的父序列 (意思是，一个最短的序列使得所有 `seqs`里的序列都是它的子序列).  
判断是否有且仅有一个能从 `seqs`重构出来的序列，并且这个序列是`org`。

例1:
```python
输入:org = [1,2,3], seqs = [[1,2],[1,3]]
输出: false
解释:
[1,2,3] 并不是唯一可以被重构出的序列，还可以重构出 [1,3,2]
```

例2:
```python
输入: org = [1,2,3], seqs = [[1,2]]
输出: false
解释:
能重构出的序列只有 [1,2]，无法重构出 [1,2,3]
```

例3:
```python
输入: org = [1,2,3], seqs = [[1,2],[1,3],[2,3]]
输出: true
解释:
序列 [1,2], [1,3], 和 [2,3] 可以唯一重构出 [1,2,3].
```

例4:
```python
输入:org = [4,1,5,2,6,3], seqs = [[5,2,6,3],[4,1,5,2]]
输出:true
```

---

### 解法：BFS（拓扑排序）

#### 思路

1. **构建图和入度数组**：
    
    - 使用邻接表构建有向图，表示顺序依赖关系。
    - 计算每个节点的入度，表示需要多少前置节点完成才能访问该节点。
2. **初始化队列**：
    
    - 将所有入度为 `0` 的节点加入队列。
    - 如果同时有多个入度为 `0` 的节点，说明序列不唯一，直接返回 `False`。
3. **BFS 遍历**：
    
    - 每次从队列中取出一个节点，将其加入重建的序列中。
    - 遍历该节点的邻居，将邻居节点的入度减 `1`。
    - 如果某个邻居节点的入度减为 `0`，加入队列。
    - 如果同时有多个入度为 `0` 的节点，说明序列不唯一，直接返回 `False`。
4. **验证结果**：
    
    - BFS 遍历完成后，检查重建的序列是否与 `org` 相同。
    - 如果不同，返回 `False`；否则返回 `True`。

---
Example:
例1:
```python
输入:org = [1,2,3], seqs = [[1,2],[1,3]]
输出: false
解释:
[1,2,3] 并不是唯一可以被重构出的序列，还可以重构出 [1,3,2]
```
例2:
```python
输入: org = [1,2,3], seqs = [[1,2]]
输出: false
解释:
能重构出的序列只有 [1,2]，无法重构出 [1,2,3]
```
例3:
```python
输入: org = [1,2,3], seqs = [[1,2],[1,3],[2,3]]
输出: true
解释:
序列 [1,2], [1,3], 和 [2,3] 可以唯一重构出 [1,2,3].
```
例4:
```python
输入:org = [4,1,5,2,6,3], seqs = [[5,2,6,3],[4,1,5,2]]
输出:true
```


#### 代码实现

```python
class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """
    def sequenceReconstruction(self, org, seqs):
        graph = self.build_graph(seqs)
        topo_order = self.topological_sort(graph)
        return topo_order == org
            
    def build_graph(self, seqs):
        # initialize graph
        graph = {}
        for seq in seqs:
            for node in seq:
                if node not in graph:
                    graph[node] = set()
        
        for seq in seqs:
            for i in range(1, len(seq)):
                graph[seq[i - 1]].add(seq[i])

        return graph
    
    def get_indegrees(self, graph):
        indegrees = {
            node: 0
            for node in graph
        }
        
        for node in graph:
            for neighbor in graph[node]:
                indegrees[neighbor] += 1
                
        return indegrees
        
    def topological_sort(self, graph):
        indegrees = self.get_indegrees(graph)
        
        queue = []
        for node in graph:
            if indegrees[node] == 0:
                queue.append(node)
        
        topo_order = []
        while queue:
            if len(queue) > 1:
                # there must exist more than one topo orders
                return None
                
            node = queue.pop()
            topo_order.append(node)
            for neighbor in graph[node]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(topo_order) == len(graph):
            return topo_order
            
        return None

org = [1,2,3], 
seqs = [[1,2],[1,3],[2,3]]
sol = Solution()
output = sol.sequenceReconstruction(org, seqs)
print(output)
```
pass

### 方法简介

这段代码通过构建有向图并执行 **拓扑排序（Topological Sorting）** 来验证是否可以从 `seqs` 唯一重建目标序列 `org`。分为以下几个步骤：

1. 构建有向图，表示序列间的依赖关系。
2. 计算图中每个节点的入度。
3. 执行拓扑排序：
    - 验证是否存在且仅存在唯一的拓扑排序。
4. 将生成的拓扑排序与 `org` 比较，若相同则返回 `True`，否则返回 `False`。

---

### 代码逐步解析

#### 主函数：`sequenceReconstruction`
```python
def sequenceReconstruction(self, org, seqs):
    graph = self.build_graph(seqs)          # 步骤 1: 构建有向图
    topo_order = self.topological_sort(graph)  # 步骤 2: 执行拓扑排序
    return topo_order == org               # 步骤 3: 验证拓扑排序是否唯一且与 org 相同

```


**逻辑解析**：

1. `build_graph`：根据 `seqs` 构建图，其中节点表示数字，边表示数字间的先后关系。
2. `topological_sort`：执行拓扑排序，同时验证是否存在唯一的拓扑排序。
3. 比较生成的拓扑序列与 `org` 是否完全一致。

---

#### 函数 1：`build_graph`
```python
def build_graph(self, seqs):
    # 初始化图
    graph = {}
    for seq in seqs:
        for node in seq:
            if node not in graph:
                graph[node] = set()
    
    # 构建有向图的边
    for seq in seqs:
        for i in range(1, len(seq)):
            graph[seq[i - 1]].add(seq[i])

    return graph

```


**步骤解析**：

1. **初始化图**：
    
    - 遍历 `seqs` 中的每个子序列，确保每个节点都出现在图中，即使没有任何边。
    - 例如，`seqs = [[1, 2], [3]]`，图初始化为 `{1: set(), 2: set(), 3: set()}`。
2. **添加边**：
    
    - 对于每个子序列中的相邻节点，构建一条从前一个节点指向后一个节点的边。
    - 例如，`seqs = [[1, 2], [2, 3]]`，最终图为：
        
  
```python
graph = {
    1: {2},
    2: {3},
    3: set()
}

```
        

---

#### 函数 2：`get_indegrees`
```python
def get_indegrees(self, graph):
    indegrees = {node: 0 for node in graph}  # 初始化每个节点的入度为 0
    
    for node in graph:
        for neighbor in graph[node]:
            indegrees[neighbor] += 1         # 对所有邻居节点增加入度
            
    return indegrees

```

**步骤解析**：

1. **初始化入度数组**：
    
    - 将图中每个节点的初始入度设置为 0。
2. **计算入度**：
    
    - 遍历图中的每条边，对于每个节点的邻居，将其入度加 1。

**示例**：

- 图：`{1: {2}, 2: {3}, 3: set()}`
- 计算入度后：`indegrees = {1: 0, 2: 1, 3: 1}`。

---

#### 函数 3：`topological_sort`
```python
def topological_sort(self, graph):
    indegrees = self.get_indegrees(graph)  # 计算入度
    
    queue = []
    for node in graph:
        if indegrees[node] == 0:          # 找到所有入度为 0 的节点
            queue.append(node)
    
    topo_order = []
    while queue:
        if len(queue) > 1:               # 队列中存在多个节点，拓扑排序不唯一
            return None
            
        node = queue.pop()               # 弹出队列中的节点
        topo_order.append(node)
        
        for neighbor in graph[node]:     # 遍历邻居节点
            indegrees[neighbor] -= 1
            if indegrees[neighbor] == 0: # 若邻居入度变为 0，则加入队列
                queue.append(neighbor)
    
    if len(topo_order) == len(graph):    # 检查是否完成拓扑排序
        return topo_order
    
    return None                          # 存在环，返回 None

```

**步骤解析**：

1. **初始化队列**：
    
    - 将所有入度为 0 的节点加入队列，表示它们可以直接被访问。
2. **BFS 遍历**：
    
    - 每次从队列中取出一个节点，将其加入拓扑排序结果。
    - 遍历其邻居节点，将邻居的入度减 1，如果邻居的入度变为 0，加入队列。
    - 如果队列中有多个节点，说明当前图的拓扑排序不唯一，直接返回 `None`。
3. **验证结果**：
    
    - 如果拓扑排序的结果包含图中所有节点，返回结果；否则，返回 `None`。

---

#### 整体逻辑示例

**输入**：
```python
org = [1, 2, 3]
seqs = [[1, 2], [2, 3]]

```

**运行过程**：

1. **构建图**：
```python
graph = {
    1: {2},
    2: {3},
    3: set()
}

```
    
2. **计算入度**：

    `indegrees = {     1: 0,     2: 1,     3: 1 }`
    
3. **拓扑排序**：
    
    - 初始队列：`queue = [1]`
    - 第 1 步：`1 -> topo_order = [1]`，更新队列：`queue = [2]`。
    - 第 2 步：`2 -> topo_order = [1, 2]`，更新队列：`queue = [3]`。
    - 第 3 步：`3 -> topo_order = [1, 2, 3]`，队列为空。
4. **验证结果**：
    
    - `topo_order == org`，返回 `True`。

---

### 时间和空间复杂度

#### 时间复杂度：

1. **构建图**：遍历 `seqs`，复杂度为 `O(E)`，其中 `E` 是 `seqs` 中的边数。
2. **计算入度**：遍历图中的边，复杂度为 `O(E)`。
3. **拓扑排序**：遍历所有节点和边，复杂度为 `O(V + E)`。

总时间复杂度：`O(V + E)`，其中 `V` 是节点数，`E` 是边数。

#### 空间复杂度：

1. 图和入度数组占用 `O(V + E)`。
2. 队列最多存储 `O(V)` 个节点。

总空间复杂度：`O(V + E)`。

---

### 其他解法

1. **DFS 拓扑排序**：
    
    - 使用深度优先搜索构建拓扑排序，同时检测是否存在环。
    - 如果递归过程中发现多个分支，说明排序不唯一。
2. **逐段验证法**：
    
    - 遍历 `seqs`，逐段验证是否符合 `org` 的顺序约束。

---

### 总结

这段代码基于 **图的构建** 和 **拓扑排序** 来判断序列是否唯一重建。代码结构清晰、易于理解，适用于包含大量依赖关系的场景。拓扑排序在唯一性验证中非常高效，时间复杂度为 `O(V + E)`。