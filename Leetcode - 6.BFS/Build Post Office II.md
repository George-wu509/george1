
### LintCode 573: 邮局的建立 II (Build Post Office II)

---

### 问题描述

给出一个二维的网格，每一格可以代表墙 `2` ，房子 `1`，以及空 `0` (用数字0,1,2来表示)，在网格中找到一个位置去建立邮局，使得所有的房子到邮局的距离和是最小的。  
返回所有房子到邮局的最小距离和，如果没有地方建立邮局，则返回`-1`.

```python
输入：[
[1,0,2,0,1],
[0,0,0,0,0],
[0,0,1,0,0] ]
输出：4
```

---

### 解法：BFS（广度优先搜索）

#### 思路

1. **多源 BFS**：
    
    - 对每个居民楼作为起点，进行 BFS，计算从当前居民楼到所有空地的距离。
    - 累积所有居民楼到每个空地的距离。
2. **累积距离**：
    
    - 使用一个二维数组 `dist` 存储每个空地的总距离。
    - 使用另一个二维数组 `reachable` 记录每个空地可以被多少个居民楼访问。
3. **结果计算**：
    
    - 遍历所有空地，找到可以被所有居民楼访问且距离最小的位置。
    - 如果没有空地可以访问所有居民楼，返回 `-1`。

---

### 代码实现
```python
from collections import deque

class Solution:
    def shortestDistance(self, grid):
        if not grid or not grid[0]:
            return -1

        rows, cols = len(grid), len(grid[0])
        dist = [[0] * cols for _ in range(rows)]
        reachable = [[0] * cols for _ in range(rows)]
        total_buildings = 0

        # 遍历所有居民楼，进行 BFS
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    total_buildings += 1
                    self.bfs(grid, i, j, dist, reachable)

        # 查找可以被所有居民楼访问且距离最小的空地
        min_distance = float('inf')
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 0 and reachable[i][j] == total_buildings:
                    min_distance = min(min_distance, dist[i][j])

        return min_distance if min_distance != float('inf') else -1

    def bfs(self, grid, start_x, start_y, dist, reachable):
        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        queue = deque([(start_x, start_y, 0)])  # (x, y, 当前距离)

        while queue:
            x, y, distance = queue.popleft()

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 0:
                    visited[nx][ny] = True
                    dist[nx][ny] += distance + 1
                    reachable[nx][ny] += 1
                    queue.append((nx, ny, distance + 1))

```
pass

---

### 示例输入输出

#### 示例 1

**输入**：
```python
grid = [
    [1, 0, 2, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0]
]
```
**运行过程**：

1. **初始化**：
    
    - `dist` 和 `reachable` 初始化为全零矩阵：
```python
dist = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]
reachable = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

```
    - 总居民楼数：`total_buildings = 3`。
2. **BFS 从居民楼 (0, 0) 开始**：
    
    - 更新 `dist` 和 `reachable`：
```python
dist = [
    [0, 1, 0, 1, 0],
    [1, 2, 1, 2, 1],
    [2, 3, 2, 3, 2]
]
reachable = [
    [0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1]
]

```
        
3. **BFS 从居民楼 (0, 4) 开始**：
    
    - 更新 `dist` 和 `reachable`：
```python
dist = [
    [0, 1, 0, 1, 0],
    [1, 3, 2, 3, 1],
    [2, 4, 3, 4, 2]
]
reachable = [
    [0, 1, 0, 1, 0],
    [1, 2, 2, 2, 1],
    [1, 2, 0, 2, 1]
]

```
        
4. **BFS 从居民楼 (2, 2) 开始**：
    
    - 更新 `dist` 和 `reachable`：
```python
dist = [
    [0, 1, 0, 1, 0],
    [1, 3, 4, 3, 1],
    [2, 4, 3, 4, 2]
]
reachable = [
    [0, 1, 0, 1, 0],
    [1, 3, 3, 3, 1],
    [1, 3, 0, 3, 1]
]

```
        
5. **查找最优空地**：
    
    - 遍历 `dist` 和 `reachable`：
        - 空地 (1, 2) 的总距离最小，为 `4`。

**输出**：

`4`

---

### 时间和空间复杂度分析

#### 时间复杂度

1. **多源 BFS**：
    
    - 每个居民楼都需要进行 BFS，复杂度为 `O(B * (R * C))`，其中：
        - `B` 是居民楼数量。
        - `R` 和 `C` 分别是网格的行和列。
2. **查找最优空地**：
    
    - 遍历网格一次，复杂度为 `O(R * C)`。

总时间复杂度为 `O(B * R * C)`。

#### 空间复杂度

1. BFS 使用的队列和访问矩阵：
    - 空间复杂度为 `O(R * C)`。

总空间复杂度为 `O(R * C)`。

---

### 其他解法简述

#### 1. 暴力遍历

- 对每个空地进行 BFS，计算所有居民楼到该空地的距离。
- 时间复杂度为 `O((R * C) * (B * R * C))`，效率较低。

#### 2. 动态规划优化

- 使用两次扫描（从左上到右下、从右下到左上），动态记录每个空地的距离和可达性。
- 适合特定情况下的优化，但实现复杂。

---

### 总结

- BFS 是解决本问题的最直观且高效的方法，利用多源 BFS 累积距离。
- 时间复杂度为 `O(B * R * C)`，适合较大规模网格和居民楼数量的情况。
- 暴力解法和动态规划适合特定情景，但一般不如 BFS 高效。