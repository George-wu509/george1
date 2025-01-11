
### LintCode 616: 课程表 II（Course Schedule II）

在 `LintCode 616` 中，我们需要找到一种课程安排顺序，使得所有课程都能被完成。课程间存在依赖关系，如果无法完成所有课程，则返回空数组。

---

### 问题描述

1. 有 `numCourses` 门课程，用 `0` 到 `numCourses-1` 编号。
2. 一个数组 `prerequisites`，其中 `prerequisites[i] = [a, b]` 表示要学习课程 `a`，你必须先完成课程 `b`。

**目标**：

- 如果可以完成所有课程，返回一种课程安排的顺序。
- 如果无法完成所有课程，返回空数组。

---

### 解法：BFS（拓扑排序）

我们使用 **拓扑排序（Topological Sorting）** 的方法来解决问题。这是一种常见的用于有向无环图（DAG）的算法，可以用 BFS 实现。

#### 思路

1. **建图**：
    
    - 使用邻接表表示课程之间的依赖关系。
    - 计算每门课程的入度（indegree），即需要先修的课程数。
2. **初始化队列**：
    
    - 将所有入度为 `0` 的课程加入队列，表示这些课程可以直接学习。
3. **BFS 遍历**：
    
    - 从队列中取出课程，将其加入课程顺序（结果数组）。
    - 遍历其依赖的课程，将这些课程的入度减 `1`。
    - 如果某课程的入度减为 `0`，将其加入队列。
4. **终止条件**：
    
    - 如果结果数组的大小等于课程总数，说明可以完成所有课程，返回结果。
    - 否则，返回空数组，表示无法完成所有课程（存在环）。

---
Example:
**例1:**
```
输入: n = 2, prerequisites = [[1,0]] 
输出: [0,1]
```
**例2:**
```
输入: n = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]] 
输出: [0,1,2,3] or [0,2,1,3]
```


#### 代码实现

```python
from collections import deque, defaultdict

def find_order(num_courses, prerequisites):
    # 建图和入度数组
    graph = defaultdict(list)
    indegree = [0] * num_courses

    for a, b in prerequisites:
        graph[b].append(a)
        indegree[a] += 1

    # 初始化队列
    queue = deque([i for i in range(num_courses) if indegree[i] == 0])
    result = []

    # BFS 遍历
    while queue:
        course = queue.popleft()
        result.append(course)

        # 遍历当前课程的邻居
        for neighbor in graph[course]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # 检查是否完成所有课程
    return result if len(result) == num_courses else []

```
pass

#### 示例输入输出

##### 示例 1

**输入**：

`numCourses = 4 prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]`

**运行过程**：

1. **建图和计算入度**：
    

    `graph = {     0: [1, 2],     1: [3],     2: [3],     3: [] } indegree = [0, 1, 1, 2]`
    
2. **初始化队列**：
    
    - 入度为 `0` 的课程：`queue = deque([0])`。
3. **BFS 遍历**：
    
    - 第 1 步：取出课程 `0`，`result = [0]`。
        
        - 课程 `1` 入度减为 `0`，加入队列。
        - 课程 `2` 入度减为 `0`，加入队列。
        - 更新：`queue = deque([1, 2])`，`indegree = [0, 0, 0, 2]`。
    - 第 2 步：取出课程 `1`，`result = [0, 1]`。
        
        - 课程 `3` 入度减为 `1`，未加入队列。
        - 更新：`queue = deque([2])`，`indegree = [0, 0, 0, 1]`。
    - 第 3 步：取出课程 `2`，`result = [0, 1, 2]`。
        
        - 课程 `3` 入度减为 `0`，加入队列。
        - 更新：`queue = deque([3])`，`indegree = [0, 0, 0, 0]`。
    - 第 4 步：取出课程 `3`，`result = [0, 1, 2, 3]`。
        
4. **结果检查**：
    
    - 所有课程都已加入结果，返回 `[0, 1, 2, 3]`。

**输出**：

`[0, 1, 2, 3]`

---

### 时间和空间复杂度分析

#### 时间复杂度：

1. **建图**：
    - 遍历 `prerequisites`，复杂度为 `O(E)`，其中 `E` 是边的数量。
2. **BFS**：
    - 每个节点（课程）最多访问一次，复杂度为 `O(V)`，其中 `V` 是节点数量。
    - 遍历所有邻居，总复杂度为 `O(E)`。

总时间复杂度为 `O(V + E)`。

#### 空间复杂度：

1. 图的邻接表需要 `O(E)` 空间。
2. 入度数组需要 `O(V)` 空间。
3. 队列和结果数组最多存储 `V` 个课程，额外空间为 `O(V)`。

总空间复杂度为 `O(V + E)`。

---

### 其他解法简述

#### 1. DFS（拓扑排序）

- 使用深度优先搜索检测是否有环并生成拓扑排序。
- 如果在递归过程中发现回到已访问节点（还未完成路径的节点），说明存在环。
- 时间复杂度同样为 `O(V + E)`，但实现较为复杂。

#### 2. Kahn 算法（BFS 的变种）

- 与 BFS 类似，但直接将节点按层次关系输出。
- Kahn 算法更适用于处理环检测。

---

### 总结

- **BFS（拓扑排序）** 是解决该问题的首选方法，易于实现且效率高。
- **DFS** 提供另一种拓扑排序方式，但代码实现更复杂。
- 两种方法的时间和空间复杂度均为 `O(V + E)`，适用于大规模课程和依赖关系的情况。