

### LintCode 433: 岛屿的个数

---

### 问题描述

给定一个 2D 网格 `grid`，其中：

- `1` 表示陆地。
- `0` 表示水域。

找出网格中**岛屿的数量**，岛屿被定义为由相邻的 `1`（水平或垂直方向）组成的区域，周围被 `0` 包围。

---

### 解法：BFS（广度优先搜索）

#### 思路

1. **网格遍历**：
    
    - 遍历整个 `grid`，当遇到一个值为 `1` 且未访问的格子时，计数器 `islands` 加 `1`。
    - 从该格子开始进行 BFS，标记整个岛屿为已访问。
2. **BFS 遍历岛屿**：
    
    - 使用队列存储待处理的格子，每次取出一个格子，并检查其四个方向的相邻格子。
    - 如果相邻格子是陆地且未访问，则加入队列。
3. **边界检查**：
    
    - 在检查相邻格子时，确保其不越界、未访问且是陆地。
4. **结果**：
    
    - BFS 完成后，所有属于同一岛屿的格子都会被标记为已访问，继续遍历下一个未访问的格子。

---
Example:
**样例 1：**
```python
输入：
[
  [1,1,0,0,0],
  [0,1,0,0,1],
  [0,0,0,1,1],
  [0,0,0,0,0],
  [0,0,0,0,1]
]
输出：
3
```
**样例 2：**
```python
输入：
[
  [1,1]
]
输出：
1
```


###代码实现
```python
from collections import deque

DIRECTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]

class Solution:
    def num_islands(self, grid):
        if not grid or not grid[0]:
            return 0
            
        islands = 0
        visited = set()  # 用于标记已访问的格子
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                # 如果当前格子是陆地且未访问，开始新的 BFS
                if grid[i][j] and (i, j) not in visited:
                    self.bfs(grid, i, j, visited)
                    islands += 1  # 岛屿计数加 1
                    
        return islands                    

    def bfs(self, grid, x, y, visited):
        queue = deque([(x, y)])
        visited.add((x, y))  # 标记起点为已访问
        
        while queue:
            x, y = queue.popleft()
            for delta_x, delta_y in DIRECTIONS:
                next_x = x + delta_x
                next_y = y + delta_y
                # 检查是否是有效的相邻格子
                if not self.is_valid(grid, next_x, next_y, visited):
                    continue
                queue.append((next_x, next_y))
                visited.add((next_x, next_y))  # 标记为已访问

    def is_valid(self, grid, x, y, visited):
        n, m = len(grid), len(grid[0])
        # 检查是否越界
        if not (0 <= x < n and 0 <= y < m):
            return False
        # 检查是否已访问
        if (x, y) in visited:
            return False
        # 检查是否是陆地
        return grid[x][y]

```
pass
解釋:
step1 在num_islands function遍歷整個grid, 如果有數值=1而且visited=0, 則bfs()  
step2 在bfs()用BFS, deque 將相鄰的格子visited=1  
step3  回到num_islands function裡對每個bfs(), islands +1  


### 示例输入输出

#### 示例输入

`grid = [   [1, 1, 0, 0, 0],   [0, 1, 0, 0, 1],   [0, 0, 0, 1, 1],   [0, 0, 0, 0, 0],   [0, 0, 0, 0, 1] ]`

#### 运行过程

1. **初始化**：
    
    - `islands = 0`
    - `visited = set()`
2. **第一次 BFS**：
    
    - 起点：`(0, 0)`。
    - 访问顺序：`(0, 0) -> (0, 1) -> (1, 1)`。
    - 标记访问：`visited = {(0, 0), (0, 1), (1, 1)}`。
    - 岛屿计数：`islands = 1`。
3. **第二次 BFS**：
    
    - 起点：`(1, 4)`。
    - 访问顺序：`(1, 4) -> (2, 4) -> (2, 3)`。
    - 标记访问：`visited = {(0, 0), (0, 1), (1, 1), (1, 4), (2, 4), (2, 3)}`。
    - 岛屿计数：`islands = 2`。
4. **第三次 BFS**：
    
    - 起点：`(4, 4)`。
    - 访问顺序：`(4, 4)`。
    - 标记访问：`visited = {(0, 0), (0, 1), (1, 1), (1, 4), (2, 4), (2, 3), (4, 4)}`。
    - 岛屿计数：`islands = 3`。
5. **结果**：
    
    - 返回岛屿数量：`3`。

**输出**：

python

複製程式碼

`3`

---

### 时间和空间复杂度分析

#### 时间复杂度

1. **遍历网格**：
    
    - 每个格子最多被访问一次。
    - 总复杂度为 `O(N * M)`，其中 `N` 是行数，`M` 是列数。
2. **BFS**：
    
    - 每个岛屿的所有格子在 BFS 中被访问一次。
    - 总复杂度为 `O(N * M)`。

总时间复杂度为 **`O(N * M)`**。

#### 空间复杂度

1. **队列和访问记录**：
    - 队列最大存储一个岛屿的所有格子，空间复杂度为 `O(L)`，其中 `L` 是岛屿的最大面积。
    - 访问记录 `visited` 的空间复杂度为 `O(N * M)`。

总空间复杂度为 **`O(N * M)`**。

---

### 其他解法简述

#### 1. DFS（深度优先搜索）

- 使用递归代替队列完成深度优先搜索。
- 时间复杂度和空间复杂度与 BFS 相同，但递归深度可能受限。

#### 2. 并查集（Union-Find）

- 将每个陆地格子视为节点，连接相邻陆地。
- 最终统计联通分量的数量。
- 时间复杂度为 `O(N * M * α(N * M))`，`α` 是阿克曼函数的反函数，近似常数。

---

### 总结

- **BFS** 是解决此问题的经典方法，代码直观，适用于所有规模的网格。
- 时间复杂度和空间复杂度均为 **`O(N * M)`**。
- **DFS** 和 **并查集** 提供了不同的实现方式，适合不同场景的优化需求。
