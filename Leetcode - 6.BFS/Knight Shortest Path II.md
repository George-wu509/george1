


### LintCode 630: 骑士的最短路径（二）

---

### 问题描述

给定一个棋盘 `grid`，棋盘大小为 `n x m`，骑士从左上角 `(0, 0)` 出发，目标是到达右下角 `(n-1, m-1)`。棋盘中 `1` 表示障碍物，`0` 表示可以经过的格子。每次移动骑士只能按照以下规则移动：

- 向前跳 2 步，再左右跳 1 步。
- 向后跳 2 步，再左右跳 1 步。

要求返回骑士到达目标的最短步数，如果无法到达则返回 `-1`。

---

### 解法：双向 BFS（双向广度优先搜索）

#### 思路

1. **问题转化**：
    
    - 以 `(0, 0)` 为起点，从左上角开始搜索。
    - 以 `(n-1, m-1)` 为终点，从右下角开始搜索。
    - 当两个搜索队列相遇时，即找到最短路径。
2. **双向 BFS 的优点**：
    
    - 同时从起点和终点开始搜索，减少搜索空间。
    - 如果两个搜索队列相遇，路径长度为两侧搜索步数之和。
3. **核心实现**：
    
    - 分别用 `forward_queue` 和 `backward_queue` 表示从起点和终点的搜索队列。
    - 使用 `visited` 集合记录已访问的节点，避免重复计算。
    - 使用方向数组分别定义前向和后向的骑士移动规则。
4. **终止条件**：
    
    - 如果两个队列相遇，则返回路径长度。
    - 如果队列为空，则返回 `-1`，表示目标不可达。

---
Example:
例1:
```
输入:
[[0,0,0,0],[0,0,0,0],[0,0,0,0]]
输出:
3
解释:
[0,0]->[2,1]->[0,2]->[2,3]
```
例2:
```
输入:
[[0,1,0],[0,0,1],[0,0,0]]
输出:
-1
```


### 代码实现
```python
import collections

FORWARD_DIRECTIONS = (
    (1, 2),
    (-1, 2),
    (2, 1),
    (-2, 1),
)

BACKWARD_DIRECTIONS = (
    (-1, -2),
    (1, -2),
    (-2, -1),
    (2, -1),
)

class Solution:
    def shortest_path2(self, grid):
        if not grid or not grid[0]:
            return -1
            
        n, m = len(grid), len(grid[0])
        if grid[n - 1][m - 1]:  # 目标点为障碍物
            return -1
        if n * m == 1:  # 起点即终点
            return 0
            
        forward_queue = collections.deque([(0, 0)])
        forward_set = set([(0, 0)])
        backward_queue = collections.deque([(n - 1, m - 1)])
        backward_set = set([(n - 1, m - 1)])
        
        distance = 0
        while forward_queue and backward_queue:
            distance += 1
            if self.extend_queue(forward_queue, FORWARD_DIRECTIONS, forward_set, backward_set, grid):
                return distance
                
            distance += 1
            if self.extend_queue(backward_queue, BACKWARD_DIRECTIONS, backward_set, forward_set, grid):
                return distance

        return -1
                
    def extend_queue(self, queue, directions, visited, opposite_visited, grid):
        for _ in range(len(queue)):
            x, y = queue.popleft()
            for dx, dy in directions:
                new_x, new_y = (x + dx, y + dy)
                if not self.is_valid(new_x, new_y, grid, visited):
                    continue
                if (new_x, new_y) in opposite_visited:
                    return True
                queue.append((new_x, new_y))
                visited.add((new_x, new_y))
                
        return False
        
    def is_valid(self, x, y, grid, visited):
        if x < 0 or x >= len(grid):
            return False
        if y < 0 or y >= len(grid[0]):
            return False
        if grid[x][y]:  # 障碍物
            return False
        if (x, y) in visited:  # 已访问
            return False
        return True

```
pass


### 示例输入输出

#### 输入

`grid = [     [0, 0, 0, 0],     [0, 0, 0, 0],     [0, 0, 0, 0] ]`

#### 运行过程

1. **初始化**：
    
    - `forward_queue = deque([(0, 0)])`
    - `backward_queue = deque([(2, 3)])`
    - `distance = 0`
2. **第一轮 BFS**：
    
    - **前向搜索**：
        - 起点 `(0, 0)` 可移动到 `(1, 2)`、`2, 1`。
        - 更新队列：`forward_queue = deque([(1, 2), (2, 1)])`。
        - 更新访问：`forward_set = {(0, 0), (1, 2), (2, 1)}`。
    - **后向搜索**：
        - 终点 `(2, 3)` 可移动到 `(1, 1)`、`0, 2`。
        - 更新队列：`backward_queue = deque([(1, 1), (0, 2)])`。
        - 更新访问：`backward_set = {(2, 3), (1, 1), (0, 2)}`。
3. **第二轮 BFS**：
    
    - **前向搜索**：
        - 起点方向搜索队列的节点 `1,2`，发现 `1,1` 在 `backward_set` 中。
        - 前向搜索与后向搜索相遇，返回路径长度 `2`。

#### 输出

python

複製程式碼

`2`

---

### 时间和空间复杂度分析

#### 时间复杂度

1. **BFS**：
    
    - 每个节点最多访问一次，每次最多尝试 4 个方向。
    - 总复杂度为 `O(N * M)`，其中 `N` 是网格行数，`M` 是网格列数。
2. **双向 BFS**：
    
    - 搜索空间减半，总复杂度仍为 `O(N * M)`。

#### 空间复杂度

1. **队列和访问记录**：
    - `visited` 和队列最多存储 `O(N * M)` 个节点。

总空间复杂度为 **`O(N * M)`**。

---

### 其他解法简述

#### 1. 单向 BFS

- 从起点 `(0, 0)` 出发，使用普通 BFS 搜索到终点 `(n-1, m-1)`。
- 时间复杂度和空间复杂度均为 `O(N * M)`，但搜索空间较大。

#### 2. A* 搜索

- 使用启发式函数（如曼哈顿距离）优化搜索顺序。
- 在路径较长或障碍较多时性能更优，但实现复杂度较高。

---

### 总结

- **双向 BFS** 是解决此问题的最佳选择，搜索空间小，适合大规模棋盘。
- 时间复杂度和空间复杂度均为 **`O(N * M)`**。
- **单向 BFS** 和 **A*** 提供了替代方案，适用于不同规模和约束的场景。


