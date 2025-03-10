
### LintCode 1565: 飞行棋 I

---

### 问题描述

在一个长度为 `length` 的直线上有飞行棋盘，棋盘上可能存在一些特殊的传送门（`connections`），如果棋子到达一个传送门的起点位置，它可以直接到达传送门的终点位置。每次掷骰子可以前进 1 到 6 格。问从起点位置 `1` 到终点位置 `length`，所需的最少步数。

Example:
**样例1**
```python
输入: length = 10 和 connections = [[2, 10]]
输出: 1
解释: 
1->2 (投骰子)
2->10(直接相连)
```
**样例2**
```python
输入: length = 15 和 connections = [[2, 8], [6, 9]]
输出: 2
解释: 
1->6 (投骰子)
6->9 (直接相连)
9->15(投骰子)
```


---

### 解法：BFS（广度优先搜索）

#### 思路

1. **构建图**：
    
    - 将棋盘上的传送门表示为一个有向图。
    - 如果某个位置 `a` 有传送门连接到位置 `b`，则从节点 `a` 到 `b` 存在一条有向边。
2. **状态定义**：
    
    - `distance[node]` 表示从起点位置 `1` 到达位置 `node` 的最小步数。
3. **BFS 搜索**：
    
    - 从起点位置 `1` 开始，使用队列记录待访问的节点。
    - 每次从队列中取出一个节点，尝试掷骰子前进 1 到 6 步。
    - 如果前进后的节点有传送门，则通过传送门到达相应的终点位置。
4. **终止条件**：
    
    - 如果某一轮搜索中到达终点位置 `length`，返回当前步数。


### 代码实现
```python
from collections import deque

class Solution:
    """
    @param length: the length of board
    @param connections: the connections of the positions
    @return: the minimum steps to reach the end
    """
    def modern_ludo(self, length, connections):
        # 构建传送门图
        graph = self.build_graph(length, connections)

        # BFS 初始化
        queue = deque([1])
        distance = {1: 0}  # 起点的距离为 0

        # BFS 遍历
        while queue:
            node = queue.popleft()

            # 掷骰子前进 1 到 6 格
            for neighbor in range(node + 1, min(node + 7, length + 1)):
                # 获取从当前节点可到达的未访问节点
                connected_nodes = self.get_unvisited_nodes(graph, distance, neighbor)
                for connected_node in connected_nodes:
                    # 更新步数并加入队列
                    distance[connected_node] = distance[node] + 1
                    queue.append(connected_node)

        return distance[length]  # 返回终点的最小步数

    def build_graph(self, length, connections):
        # 构建传送门图
        graph = {i: set() for i in range(1, length + 1)}
        for a, b in connections:
            graph[a].add(b)
        return graph

    def get_unvisited_nodes(self, graph, distance, node):
        # BFS 查找所有从 `node` 出发可以到达的未访问节点
        queue = deque([node])
        unvisited_nodes = set()
        while queue:
            node = queue.popleft()
            if node in distance:
                continue
            unvisited_nodes.add(node)
            for neighbor in graph[node]:
                if neighbor not in distance:
                    queue.append(neighbor)
                    unvisited_nodes.add(neighbor)
        return unvisited_nodes

```
pass
解釋:


---

### 示例输入输出

#### 输入

`length = 10 connections = [[3, 7], [7, 9]]`

#### 输出

`2`

#### 运行过程

1. **构建图**：
    
    - `graph = {1: set(), 2: set(), 3: {7}, 4: set(), 5: set(), 6: set(), 7: {9}, 8: set(), 9: set(), 10: set()}`。
2. **初始化**：
    
    - `queue = deque([1])`。
    - `distance = {1: 0}`。
3. **第一轮 BFS**：
    
    - 当前节点：`1`。
    - 掷骰子前进 1 到 6 格：
        - 到达位置 `2, 3, 4, 5, 6, 7`。
        - 位置 `3` 有传送门，直接到达 `7`。
        - 更新：
            - `distance = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}`。
            - `queue = deque([2, 3, 4, 5, 6, 7])`。
4. **第二轮 BFS**：
    
    - 当前节点：`7`。
    - 掷骰子前进 1 到 6 格：
        - 到达位置 `8, 9, 10`。
        - 位置 `9` 有传送门，直接到达 `10`。
        - 更新：
            - `distance = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2}`。
            - `queue = deque([2, 3, 4, 5, 6, 8, 9, 10])`。
5. **结束**：
    
    - 到达终点 `10`，步数为 `2`。

---

### 时间和空间复杂度分析

#### 时间复杂度

1. **构建图**：
    
    - 遍历传送门，复杂度为 `O(C)`，其中 `C` 是传送门数量。
2. **BFS 遍历**：
    
    - 每个节点最多访问一次，每次最多尝试 6 个掷骰子结果。
    - 复杂度为 `O(6 * L)`，其中 `L` 是棋盘长度。

总时间复杂度为 **`O(C + L)`**。

#### 空间复杂度

1. **图存储**：
    - 存储所有传送门，复杂度为 `O(C)`。
2. **访问记录**：
    - 存储所有节点的距离，复杂度为 `O(L)`。

总空间复杂度为 **`O(C + L)`**。

---

### 其他解法简述

#### 1. 优化 BFS

- 合并传送门后的节点图，减少多余的搜索。
- 时间复杂度与当前解法相同，但实际运行更快。

#### 2. 动态规划

- 使用数组 `dp[i]` 表示到达位置 `i` 的最小步数。
- 转移方程：
    
    python
    
    複製程式碼
    
    `dp[i] = min(dp[j] + 1)  # j 是 i 的前 6 个位置及其传送门`
    
- 时间复杂度为 `O(L * 6)`。

---

### 总结

- **BFS 解法**直观高效，适合求解最短路径问题。
- 时间复杂度为 **`O(C + L)`**，适合中小规模的棋盘。
- **动态规划**提供了另一种解法，适合需要记录路径的场景。