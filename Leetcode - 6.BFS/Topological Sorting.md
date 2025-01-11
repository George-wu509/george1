
### LintCode 127: 拓扑排序

---

### 问题描述

给定一个有向图，返回其任意一个拓扑排序。

**拓扑排序**定义：

- 拓扑排序是对一个有向无环图（DAG）的顶点排序，使得每条有向边 `(u, v)` 中的顶点 `u` 都排在 `v` 前面。

---

### 解法：BFS（广度优先搜索）

#### 思路

1. **入度概念**：
    
    - 入度是指一个节点有多少条边指向它。
    - 拓扑排序的起点是所有入度为 0 的节点。
2. **算法步骤**：
    
    1. **计算入度**：
        - 遍历图中所有节点，统计每个节点的入度。
    2. **初始化队列**：
        - 将所有入度为 0 的节点加入队列，这些节点是拓扑排序的起始点。
    3. **BFS 遍历**：
        - 从队列中取出一个节点，加入拓扑排序结果。
        - 遍历该节点的邻居节点，减少邻居节点的入度。
        - 如果邻居节点的入度变为 0，加入队列。
    4. **终止条件**：
        - 当队列为空时，完成排序。
3. **注意事项**：
    
    - 如果图中存在环，则无法生成拓扑排序。

---
Example:
输入：
```
graph = {0,1,2,3#1,4#2,4,5#3,4,5#4#5}
```
输出：
```
[0, 1, 2, 3, 4, 5]
```
解释：
图如下所示:

![91cf07d2-b7ea-11e9-bb77-0242ac110002.jpg](https://media-cn.lintcode.com/new_storage_v2/public/202211/6da9543e-4e23-4ff1-a33e-e99f380c1b40.jpg)

拓扑排序可以为:  
[0, 1, 2, 3, 4, 5]  
[0, 2, 3, 1, 5, 4]  
...  
您只需要返回给定图的任何一种拓扑顺序。


### 代码实现

```python
import collections

class Solution:
    """
    @param graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """
    def topSort(self, graph):
        # 计算所有节点的入度
        node_to_indegree = self.get_indegree(graph)

        # BFS 初始化
        order = []
        start_nodes = [node for node in graph if node_to_indegree[node] == 0]
        queue = collections.deque(start_nodes)

        # BFS 遍历
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in node.neighbors:
                node_to_indegree[neighbor] -= 1
                if node_to_indegree[neighbor] == 0:
                    queue.append(neighbor)
                
        return order
    
    def get_indegree(self, graph):
        # 初始化所有节点的入度为 0
        node_to_indegree = {node: 0 for node in graph}

        # 遍历所有边，计算入度
        for node in graph:
            for neighbor in node.neighbors:
                node_to_indegree[neighbor] += 1
                
        return node_to_indegree

```
pass

### 示例输入输出

#### 输入

`graph = {     0: Node(0, [1, 2, 3]),     1: Node(1, [4]),     2: Node(2, [4, 5]),     3: Node(3, [4, 5]),     4: Node(4, []),     5: Node(5, []) }`

#### 运行过程

1. **计算入度**：
    
    - 节点 `0` 的邻居：`1, 2, 3`。
        - 入度：`1->1, 2->1, 3->1`。
    - 节点 `1` 的邻居：`4`。
        - 入度：`4->1`。
    - 节点 `2` 的邻居：`4, 5`。
        - 入度：`4->2, 5->1`。
    - 节点 `3` 的邻居：`4, 5`。
        - 入度：`4->3, 5->2`。
    
    **最终入度**：
    
    python
    
    複製程式碼
    
    `node_to_indegree = {0: 0, 1: 1, 2: 1, 3: 1, 4: 3, 5: 2}`
    
2. **初始化队列**：
    
    - 入度为 0 的节点：`[0]`。
    - `queue = deque([0])`。
3. **BFS 遍历**：
    
    - 第 1 步：
        - 取出节点 `0`。
        - 更新入度：`1->0, 2->0, 3->0`。
        - 队列：`queue = deque([1, 2, 3])`。
    - 第 2 步：
        - 取出节点 `1`。
        - 更新入度：`4->2`。
        - 队列：`queue = deque([2, 3])`。
    - 第 3 步：
        - 取出节点 `2`。
        - 更新入度：`4->1, 5->1`。
        - 队列：`queue = deque([3])`。
    - 第 4 步：
        - 取出节点 `3`。
        - 更新入度：`4->0, 5->0`。
        - 队列：`queue = deque([4, 5])`。
    - 第 5 步：
        - 取出节点 `4`。
        - 无邻居。
        - 队列：`queue = deque([5])`。
    - 第 6 步：
        - 取出节点 `5`。
        - 无邻居。
        - 队列：`queue = deque([])`。
4. **输出结果**：
    
    - 拓扑排序：`[0, 1, 2, 3, 4, 5]`。

---

### 时间和空间复杂度分析

#### 时间复杂度

1. **计算入度**：
    
    - 遍历每个节点及其邻居，复杂度为 `O(V + E)`，其中 `V` 是节点数，`E` 是边数。
2. **BFS 遍历**：
    
    - 每个节点和边最多被访问一次，复杂度为 `O(V + E)`。

总时间复杂度为 **`O(V + E)`**。

#### 空间复杂度

1. **队列**：
    
    - 最多存储 `O(V)` 个节点。
2. **入度表**：
    
    - 存储所有节点的入度，复杂度为 `O(V)`。

总空间复杂度为 **`O(V)`**。

---

### 其他解法简述

#### 1. DFS（深度优先搜索）

- 使用递归对图进行深度优先遍历，记录每个节点的完成时间。
- 按完成时间倒序即为拓扑排序。
- 时间复杂度和空间复杂度均为 `O(V + E)`。

#### 2. Kahn 算法

- Kahn 算法是 BFS 的变种，基于节点的入度计算。
- 每次移除入度为 0 的节点并更新入度表。
- 本质上与本解法类似。

---

### 总结

- **BFS 解法**通过入度计算实现拓扑排序，代码清晰，适合初学者。
- 时间复杂度为 **`O(V + E)`**，空间复杂度为 **`O(V)`**，适用于中大规模图。
- **DFS 解法**提供了另一种递归思路，可在需要按时间顺序处理任务时使用。
