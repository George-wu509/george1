
### LintCode 794: 滑动拼图 II (Sliding Puzzle II)

---

### 问题描述

给定一个二维的滑动拼图网格 `start` 和一个目标状态 `target`，每次可以将空格（`0`）与其相邻的格子交换。求将 `start` 转换为 `target` 的最少步数。如果无法转换，返回 `-1`。

---

### 解法：BFS（广度优先搜索 + 队列）

#### 思路

1. **状态表示**：
    
    - 使用字符串表示拼图状态，将二维网格转换为一维字符串。
    - 例如：`[[1, 2, 3], [4, 0, 5]]` 转换为 `"123405"`。
2. **广度优先搜索**：
    
    - 从起始状态 `start` 开始，每次将 `0` 与相邻位置交换，生成新的状态。
    - 使用队列进行层次遍历，记录每个状态的步数。
    - 使用集合 `visited` 记录已访问过的状态，避免重复计算。
3. **目标检测**：
    
    - 在搜索过程中，如果找到目标状态 `target`，返回步数。
    - 如果队列为空仍未找到目标状态，返回 `-1`。
4. **转换与合法性检查**：
    
    - 对于每个状态，计算 `0` 的合法移动位置，确保生成的状态有效。

---
Example:
**样例1 ：**

```python
输入:
[
 [2,8,3],
 [1,0,4],
 [7,6,5]
]
[
 [1,2,3],
 [8,0,4],
 [7,6,5]
]
输出:
4

解释:
[                 [
 [2,8,3],          [2,0,3],
 [1,0,4],   -->    [1,8,4],
 [7,6,5]           [7,6,5]
]                 ]

[                 [
 [2,0,3],          [0,2,3],
 [1,8,4],   -->    [1,8,4],
 [7,6,5]           [7,6,5]
]                 ]

[                 [
 [0,2,3],          [1,2,3],
 [1,8,4],   -->    [0,8,4],
 [7,6,5]           [7,6,5]
]                 ]

[                 [
 [1,2,3],          [1,2,3],
 [0,8,4],   -->    [8,0,4],
 [7,6,5]           [7,6,5]
]                 ]
```

**样例 2：**

```python
输入:
[[2,3,8],[7,0,5],[1,6,4]]
[[1,2,3],[8,0,4],[7,6,5]]
输出:
-1
```


### 代码实现
```python
from typing import (

    List,

)

from typing import (
    List,
)
from collections import deque
class Solution:
    """
    @param init_state: the initial state of chessboard
    @param final_state: the final state of chessboard
    @return: return an integer, denote the number of minimum moving
    """
    def min_move_step(self, init_state: List[List[int]], final_state: List[List[int]]) -> int:
        if not init_state or not final_state:
            return -1

        # 将二维网格转为字符串
        init_state_str = self.grid_to_string(init_state)
        final_state_str = self.grid_to_string(final_state)

        # 如果起始状态就是目标状态
        if init_state_str == final_state_str:
            return 0

        # BFS 初始化
        queue = deque([(init_state_str, init_state_str.index('0'), 0)])  # (当前状态, 空格位置, 当前步数)
        visited = set()
        visited.add(init_state_str)

        # 拼图的邻居方向 (上下左右)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            current_state, zero_index, steps = queue.popleft()

            # 当前空格的行列位置
            x, y = divmod(zero_index, len(init_state[0]))

            # 遍历所有可能的移动方向
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # 检查移动是否合法
                if 0 <= nx < len(init_state) and 0 <= ny < len(init_state[0]):
                    # 计算新的空格位置
                    new_zero_index = nx * len(init_state[0]) + ny

                    # 生成新状态
                    new_state = list(current_state)
                    new_state[zero_index], new_state[new_zero_index] = new_state[new_zero_index], new_state[zero_index]
                    new_state = ''.join(new_state)

                    # 如果新状态是目标状态
                    if new_state == final_state_str:
                        return steps + 1

                    # 如果新状态未访问过
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state, new_zero_index, steps + 1))

        # 无法转换到目标状态
        return -1

    def grid_to_string(self, grid):
        return ''.join(str(cell) for row in grid for cell in row)
```
pass


### 示例输入输出

#### 示例 1

**输入**：
```python
start = [[1, 2, 3], [4, 0, 5]]
target = [[1, 2, 3], [4, 5, 0]]

```
**运行过程**：

1. **初始化**：
    
    - 起始状态：`start_str = "123405"`
    - 目标状态：`target_str = "123450"`
    - 队列：`queue = deque([("123405", 4, 0)])`
    - 已访问集合：`visited = {"123405"}`
2. **第 1 步**：
    
    - 当前状态：`"123405"`，空格位置：`4`
    - 空格的邻居为：`(3, 5)`
        - 移动 `3`：新状态 `"103425"`，加入队列。
        - 移动 `5`：新状态 `"123450"`，返回 `1`。

**输出**：

`1`

#### 示例 2

**输入**：
```python
start = [[1, 2, 3], [5, 4, 0]]
target = [[1, 2, 3], [4, 5, 0]]

```
**运行过程**：

- 无法达到目标状态，返回 `-1`。

**输出**：

`-1`

---

### 时间和空间复杂度分析

#### 时间复杂度

1. **状态空间**：
    
    - 假设网格大小为 `m x n`，共有 `(m * n)!` 种可能状态。
    - BFS 遍历每个状态，最坏情况下需要处理所有状态。
2. **邻居计算**：
    
    - 每个状态最多有 `4` 个邻居，复杂度为 `O(1)`。

总时间复杂度为 **`O((m * n)!)`**，但由于状态剪枝（避免重复访问），实际复杂度远小于理论值。

#### 空间复杂度

1. **队列和已访问集合**：
    - 最多存储所有可能状态，空间复杂度为 **`O((m * n)!)`**。

总空间复杂度为 **`O((m * n)!)`**。

---

### 其他解法简述

#### 1. 双向 BFS

- 从起始状态和目标状态同时进行 BFS，减少状态空间的搜索范围。
- 时间复杂度和空间复杂度较单向 BFS 大幅降低。

#### 2. 优先级队列（A* 搜索）

- 使用启发式函数（如曼哈顿距离）估计当前状态到目标状态的最短距离，优先扩展最有可能的状态。
- 时间复杂度依赖于启发式函数的精度。

---

### 总结

- **BFS** 是解决滑动拼图问题的经典方法，适合初学者实现，代码直观。
- **双向 BFS** 和 **A*** 提供更高效的解法，但实现复杂度略高。
- 时间和空间复杂度均为 **`O((m * n)!)`**，适合较小规模的拼图问题（如 2x3 或 3x3）。