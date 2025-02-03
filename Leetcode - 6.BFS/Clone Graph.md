
### LintCode 137: 克隆图 (Clone Graph)

---

### 问题描述

给定一个 **无向图** 的节点，要求返回该图的深度拷贝（克隆图）。无向图中的每个节点包含一个值和一个邻居列表。

**输入**：

- `node` 是图的一个节点。

**输出**：

- 返回图的克隆。

---

### 解法：BFS（广度优先搜索 + 队列）

#### 思路

1. **克隆节点和边**：
    
    - 遍历整个图，逐一复制每个节点和它的边（邻居关系）。
    - 使用一个哈希表 `node_map` 记录原节点和克隆节点的映射关系，避免重复克隆。
2. **广度优先搜索**：
    
    - 从起始节点开始，使用队列依次克隆节点及其邻居。
    - 遍历当前节点的邻居，如果邻居未克隆，则创建克隆节点并加入队列；如果已克隆，则直接添加到当前克隆节点的邻居列表。

---
Example:
```python
输入:
{1,2,4#2,1,4#4,1,2}
输出: 
{1,2,4#2,1,4#4,1,2}
解释:
1------2  
 \     |  
  \    |  
   \   |  
    \  |  
      4   
节点之间使用 '#' 分隔
1,2,4 表示某个节点 label = 1, neighbors = [2,4]
2,1,4 表示某个节点 label = 2, neighbors = [1,4]
4,1,2 表示某个节点 label = 4, neighbors = [1,2]
```


### 代码实现

```python
from collections import deque

class Solution:
    def clone_graph(self, node):
        root = node
        if node is None:
            return node
            
        # use bfs algorithm to traverse the graph and get all nodes.
        nodes = self.getNodes(node)
        
        # copy nodes, store the old->new mapping information in a hash map
        mapping = {}
        for node in nodes:
            mapping[node] = UndirectedGraphNode(node.label)
        
        # copy neighbors(edges)
        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_neighbor = mapping[neighbor]
                new_node.neighbors.append(new_neighbor)
        
        return mapping[root]
        
    def getNodes(self, node):
        q = collections.deque([node])
        result = set([node])
        while q:
            head = q.popleft()
            for neighbor in head.neighbors:
                if neighbor not in result:
                    result.add(neighbor)
                    q.append(neighbor)
        return result
```
pass

---

### 示例输入输出

#### 示例 1

**输入**：
```python
# 原图
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)

node1.neighbors = [node2, node4]
node2.neighbors = [node1, node3]
node3.neighbors = [node2, node4]
node4.neighbors = [node1, node3]

```
**运行过程**：

1. **初始化**：
    
    - 起始节点：`node1`。
    - 队列：`queue = deque([node1])`。
    - 哈希表：`node_map = {node1: Node(1)}`。
2. **第一步（处理 `node1`）**：
    
    - 弹出节点：`current_node = node1`。
    - 遍历邻居：`node2` 和 `node4`。
        - 克隆 `node2` 并加入队列：`queue = deque([node2])`，`node_map = {node1: Node(1), node2: Node(2)}`。
        - 克隆 `node4` 并加入队列：`queue = deque([node2, node4])`，`node_map = {node1: Node(1), node2: Node(2), node4: Node(4)}`。
    - 更新 `node1` 克隆节点的邻居：`node_map[node1].neighbors = [node_map[node2], node_map[node4]]`。
3. **第二步（处理 `node2`）**：
    
    - 弹出节点：`current_node = node2`。
    - 遍历邻居：`node1` 和 `node3`。
        - `node1` 已克隆，直接添加到邻居列表。
        - 克隆 `node3` 并加入队列：`queue = deque([node4, node3])`，`node_map = {node1: Node(1), node2: Node(2), node4: Node(4), node3: Node(3)}`。
    - 更新 `node2` 克隆节点的邻居：`node_map[node2].neighbors = [node_map[node1], node_map[node3]]`。
4. **第三步（处理 `node4`）**：
    
    - 弹出节点：`current_node = node4`。
    - 遍历邻居：`node1` 和 `node3`。
        - `node1` 已克隆，直接添加到邻居列表。
        - `node3` 已克隆，直接添加到邻居列表。
    - 更新 `node4` 克隆节点的邻居：`node_map[node4].neighbors = [node_map[node1], node_map[node3]]`。
5. **第四步（处理 `node3`）**：
    
    - 弹出节点：`current_node = node3`。
    - 遍历邻居：`node2` 和 `node4`。
        - `node2` 和 `node4` 已克隆，直接添加到邻居列表。
    - 更新 `node3` 克隆节点的邻居：`node_map[node3].neighbors = [node_map[node2], node_map[node4]]`。
6. **结果**：
    
    - 返回克隆的起始节点：`node_map[node1]`。

---

### 时间和空间复杂度分析

#### 时间复杂度

1. **克隆节点和边**：
    - 每个节点和边最多访问一次，复杂度为 `O(V + E)`，其中 `V` 是节点数，`E` 是边数。

#### 空间复杂度

1. **队列**：
    - 最多存储所有节点，空间复杂度为 `O(V)`。
2. **哈希表**：
    - 存储原节点和克隆节点的映射关系，空间复杂度为 `O(V)`。

总空间复杂度为 `O(V + E)`。

---

### 其他解法简述

#### 1. DFS（深度优先搜索）

- 使用递归替代队列，克隆每个节点和它的邻居。
- 时间复杂度和空间复杂度同 BFS，但递归调用可能导致栈溢出。

#### 2. 直接迭代法

- 使用栈或手动管理的队列来遍历图并克隆。
- 实现略复杂，效率与 BFS 相同。

---

### 总结

- BFS 是解决克隆图的首选方法，逻辑清晰且效率高。
- DFS 是另一种实现方式，但递归可能导致栈溢出，适用于较小的图。
- 这两种方法的时间复杂度均为 `O(V + E)`，适合大规模图的克隆操作。

4o