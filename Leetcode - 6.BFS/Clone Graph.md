
### LintCode 137: 克隆图 (Clone Graph)

---

### 问题描述

克隆一张无向图. 无向图的每个节点包含一个 `label` 和一个列表 `neighbors`. 保证每个节点的 `label` 互不相同.

你的程序需要返回一个经过深度拷贝的新图. 新图和原图具有同样的结构, 并且对新图的任何改动不会对原图造成任何影响.

**输入**：

- `node` 是图的一个节点。

**输出**：

- 返回图的克隆。

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


---

### 解法：BFS（广度优先搜索 + 队列）

#### 思路

1. **克隆节点和边**：
    
    - 遍历整个图，逐一复制每个节点和它的边（邻居关系）。
    - 使用一个哈希表 `node_map` 记录原节点和克隆节点的映射关系，避免重复克隆。
2. **广度优先搜索**：
    
    - 从起始节点开始，使用队列依次克隆节点及其邻居。
    - 遍历当前节点的邻居，如果邻居未克隆，则创建克隆节点并加入队列；如果已克隆，则直接添加到当前克隆节点的邻居列表。


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
解釋:
step1  先用getNodes() functionm用BFS將graph存到nodes是一個set 譬如 {1,2,3}  
step2  創建新node mapping = {1': UndirectedGraphNode(1'), 2': UndirectedGraphNode(2'), 3': UndirectedGraphNode(3')}  
step3  複製neighbors (edges)  最後新的node應該為
node1 = UndirectedGraphNode(1)
node2 = UndirectedGraphNode(2)
node3 = UndirectedGraphNode(3)
node1.neighbors = [node2, node3]
node2.neighbors = [node1, node3]
node3.neighbors = [node1, node2]


## **题目分析**

### **题目描述**

给定一个 **无向图**，要求 **克隆整个图**，即创建一个新图，使其结构与原始图完全相同。

### **输入**

- 一个 **无向图** 的起始 `node`，其中：
    - **每个节点都有唯一的 `label`**。
    - **每个节点都有 `neighbors`（邻接节点）**。

### **输出**

- 一个 **新的无向图**，其结构、节点和边的关系应与原始图完全相同，但**不共用内存**。

---

## **解法：BFS**

### **为什么选择 BFS？**

1. **图的遍历适用于 BFS 或 DFS**
    
    - 由于图可能是 **连通** 或 **非连通** 的，且可能包含环，BFS 适用于完整遍历整个图。
    - BFS 适用于 **层级复制**，先复制节点，再复制邻接关系，避免递归深度过深导致栈溢出。
2. **克隆图的关键**
    
    - **需要一个 `hash_map` 记录 `旧节点 → 新节点` 的映射关系**。
    - **遍历所有节点时，先创建所有 `新节点`，再复制 `邻接边`**。

---

## **解法思路**

### **步骤**

1. **获取所有节点**（BFS 遍历整个图）。
2. **复制所有节点**，并存入 `hash_map` 以建立映射关系 (`old_node → new_node`)。
3. **复制所有邻居关系**，遍历 `hash_map`，把原图的邻居关系连接到新图。

---

## **示例**


```
   1
  / \
 2 - 3
```

**图的邻接表：**

```
node1 = UndirectedGraphNode(1)
node2 = UndirectedGraphNode(2)
node3 = UndirectedGraphNode(3)
node1.neighbors = [node2, node3]
node2.neighbors = [node1, node3]
node3.neighbors = [node1, node2]
```
---

### **执行 BFS**

#### **步骤 1：遍历所有节点**

- **BFS 获取所有节点**
    - `queue = deque([1])`
    - 取出 `1`，发现邻居 `{2,3}`，入队
    - 取出 `2`，发现 `1` 和 `3`，跳过已访问的 `1`，入队 `3`
    - 取出 `3`，发现 `1` 和 `2`，全部已访问，终止
    - 结果：`{1,2,3}`

#### **步骤 2：复制所有节点**

- **创建新节点**
    - `mapping = {1': UndirectedGraphNode(1'), 2': UndirectedGraphNode(2'), 3': UndirectedGraphNode(3')}`

#### **步骤 3：复制所有邻接关系**

- 复制 `1 → [2,3]`
- 复制 `2 → [1,3]`
- 复制 `3 → [1,2]`

---

### **输出**
```
   1'
  /  \
 2' - 3'
```
---

## **时间 & 空间复杂度分析**

### **时间复杂度**

1. **获取所有节点**（BFS 遍历所有节点） → `O(N)`
2. **复制所有节点** → `O(N)`
3. **复制所有边**（遍历所有邻接表） → `O(E)`
4. **总时间复杂度**： O(N+E)
    - `N` 是图的节点数
    - `E` 是图的边数

---

### **空间复杂度**

1. **存储 `mapping` 哈希表**：`O(N)`
2. **存储 `queue`**：`O(N)`（最坏情况所有节点入队）
3. **存储 `result` 集合**：`O(N)`
4. **总空间复杂度**： O(N)

---

## **其他解法**

### **1. DFS 递归（适用于小规模图）**

- **使用递归 DFS**，每次遇到新节点就克隆，并递归克隆邻居。
- **适用于深度较小的图**，但**深度过大会导致递归栈溢出**。
- **时间复杂度**：`O(N + E)`
- **空间复杂度**：`O(N)`（递归栈）

---

### **2. DFS 迭代（显式栈）**

- **与递归 DFS 相同，但用 `stack` 显式管理遍历**，避免递归栈溢出问题。
- **时间复杂度**：`O(N + E)`
- **空间复杂度**：`O(N)`

---

## **总结**

|方法|适用情况|时间复杂度|空间复杂度|递归深度|
|---|---|---|---|---|
|**BFS（推荐）**|适用于所有图，防止栈溢出|`O(N+E)`|`O(N)`|✅ 最优|
|**DFS 递归**|适用于小规模图|`O(N+E)`|`O(N)`|❌ 深度过深会溢出|
|**DFS 迭代**|适用于避免递归栈|`O(N+E)`|`O(N)`|✅ 控制栈深|

🚀 **推荐使用 BFS 解决 Clone Graph，保证高效、无栈溢出！**