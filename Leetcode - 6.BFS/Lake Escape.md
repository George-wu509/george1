
### LintCode 1828: 湖面逃跑 (Lake Escape)

---

### 问题描述

给定一个二维网格 `grid`，其中：

- `0` 表示水域。
- `1` 表示冰面。
- `2` 表示起点。
- `3` 表示终点。

球可以从起点沿上下左右方向滑动，直到碰到边界、冰面尽头或终点为止。每次滑动消耗一单位能量，求到达终点所需的最小能量。如果无法到达终点，返回 `-1`。

---

### 解法：BFS（广度优先搜索）

#### 思路

1. **使用队列实现 BFS**：
    
    - 队列中的每个元素包含当前的坐标、滑动步数。
    - 从起点开始，每次滑动到新的位置，将新的状态加入队列。
2. **避免重复访问**：
    
    - 使用一个二维布尔数组 `visited`，记录是否访问过某个位置，避免重复搜索。
3. **滑动逻辑**：
    
    - 在当前方向上持续滑动，直到碰到水域（`0`）、边界，或到达终点（`3`）。
    - 如果到达终点，直接返回步数。
4. **终止条件**：
    
    - 队列为空时仍未到达终点，返回 `-1`。

Example:
输入:
side_length: 湖面的长度（这是一个正方形）= 7
lake_grid: 一个二维数组代表湖面，其中0代表冰面，1代表雪堆，-1代表洞
[ [0,0,0,0,0,0,0],
[0,0,-1,0,0,0,0],
[0,0,1,-1,0,-1,0],
[-1,0,1,0,0,0,0],
[0,1,1,0,0,1,0],
[-1,0,-1,0,-1,0,0],
[0,0,0,0,0,0,0] ]
albert_row: Albert所在的雪堆的行 = 4
albert_column: Albert所在的雪堆的列 = 1
kuna_row: Kuna所在的雪堆的行 = 3
kuna_column: Kuna所在的雪堆的列 = 2
输出: 
true
说明：
如图所示。黄色格子是Albert的位置，红色格子是Kuna的位置。Albert可以向右走到(4,2)，向上走到(3,2)，然后向右走，离开湖面。
![图片](https://media.jiuzhang.com/media/markdown/images/2/20/c06db360-53ec-11ea-ab9e-0242c0a8d005.jpg)

---
### LintCode 1828: 湖面逃跑

---

### 问题描述

阿尔伯特（Albert）在湖面上的一块冰上，库娜（Kuna）也在湖面上另一个位置。湖面是一个 `side_length x side_length` 的网格 `lake_grid`，其中：

- `0` 表示可以滑过的冰面。
- `-1` 表示裂开的冰面，无法滑过。
- `1` 表示有障碍物，滑行会停止。

阿尔伯特可以选择上下左右四个方向滑行，直到撞到障碍物或滑出湖面边界。目标是判断阿尔伯特能否从初始位置滑到湖面边界或者库娜所在的位置。

---

### 解法：广度优先搜索（BFS）

#### 思路

1. **问题转化**：
    
    - 以阿尔伯特的初始位置 `(albert_row, albert_column)` 为起点，模拟阿尔伯特的滑行。
    - 每次滑行时，沿着某个方向一直滑动，直到遇到：
        1. 障碍物（`1`）。
        2. 裂开的冰（`-1`）。
        3. 滑出湖面边界。
2. **状态表示**：
    
    - 状态由三元组 `(cc, x, y)` 表示，其中：
        - `cc`：是否已访问库娜位置（`0` 表示未访问，`1` 表示已访问）。
        - `(x, y)`：阿尔伯特当前位置。
3. **滑行规则**：
    
    - 如果阿尔伯特在滑行过程中经过库娜的位置，设置 `cc = 1`。
    - 如果滑出边界且 `cc = 1`，说明路径合法，返回 `True`。
4. **BFS 搜索**：
    
    - 从初始位置 `(0, albert_row, albert_column)` 开始，尝试所有可能的滑行方向。
    - 使用队列存储所有可能状态，并记录访问过的状态以避免重复搜索。
5. **终止条件**：
    
    - 如果队列为空，且没有找到合法路径，返回 `False`。

---
Example:
```
输入:
7
[[0,0,0,0,0,0,0],[0,0,-1,0,0,0,0],[0,0,1,-1,0,-1,0],[-1,0,1,0,0,0,0],[0,1,1,0,0,1,0],[-1,0,-1,0,-1,0,0],[0,0,0,0,0,0,0]]
4
1
3
2
输出: 
true
说明：
如图所示。黄色格子是Albert的位置，红色格子是Kuna的位置。Albert可以向右走到(4,2)，向上走到(3,2)，然后向右走，离开湖面。
```

### 代码实现
```python
from typing import List
from queue import Queue

class Solution:
    def lake_escape(self, side_length: int, lake_grid: List[List[int]], albert_row: int, albert_column: int, kuna_row: int, kuna_column: int) -> bool:
        visited = set()
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 四个滑行方向
        eq = Queue()
        eq.put((0, albert_row, albert_column))  # 初始状态

        while not eq.empty():
            cc, cx, cy = eq.get()

            if (cc, cx, cy) in visited:  # 避免重复搜索
                continue
            visited.add((cc, cx, cy))

            for dx, dy in directions:
                flag1 = 1  # 是否遇到裂开的冰面
                flag2 = 0  # 是否滑出边界
                flag3 = 0  # 是否经过库娜的位置
                nowx, nowy = cx, cy

                for _ in range(side_length):  # 模拟滑行
                    nowx += dx
                    nowy += dy

                    if nowx == kuna_row and nowy == kuna_column:  # 经过库娜的位置
                        flag3 = 1
                    if nowx < 0 or nowx >= side_length or nowy < 0 or nowy >= side_length:  # 滑出边界
                        flag2 = 1
                        break
                    if lake_grid[nowx][nowy] == -1:  # 遇到裂开的冰面
                        flag1 = 0
                        break
                    if lake_grid[nowx][nowy] == 1:  # 遇到障碍物，停止滑行
                        break

                if flag2 == 1:  # 滑出边界
                    if cc == 1 or flag3 == 1:  # 如果经过库娜的位置
                        return True
                elif flag1 == 1:  # 未遇到裂开的冰面
                    if flag3 == 1:
                        eq.put((1, nowx, nowy))  # 设置 cc = 1
                    else:
                        eq.put((cc, nowx, nowy))

        return False

```
pass

### 示例输入输出

#### 输入
```python
side_length = 5
lake_grid = [
    [0, 0, 0, 0, 0],
    [0, -1, -1, -1, 0],
    [0, -1, 0, -1, 0],
    [0, -1, -1, -1, 0],
    [0, 0, 0, 0, 0]
]
albert_row = 2
albert_column = 2
kuna_row = 4
kuna_column = 4

```

#### 输出

`True`

#### 运行过程

1. 初始状态：
    
    - 起点：`(0, 2, 2)`。
    - 队列：`[(0, 2, 2)]`。
2. BFS 第一轮：
    
    - 从 `(0, 2, 2)` 开始，尝试 4 个方向。
    - 滑行到 `(2, 4)`，经过库娜位置，设置 `cc = 1`。
    - 滑出边界，返回 `True`。

---

### 时间和空间复杂度分析

#### 时间复杂度

1. **滑行**：
    
    - 每次最多滑行 `side_length` 步，总共尝试 4 个方向。
    - 每个状态最多滑行 `O(side_length * 4)`。
2. **BFS 搜索**：
    
    - 最多访问所有可能状态，状态数为 `O(side_length^2)`。

总时间复杂度为 **`O(side_length^3)`**。

#### 空间复杂度

1. **队列和访问记录**：
    - 最多存储所有可能状态，空间复杂度为 `O(side_length^2)`。

总空间复杂度为 **`O(side_length^2)`**。

---

### 其他解法简述

#### 1. DFS（深度优先搜索）

- 使用递归替代队列，模拟滑行路径。
- 需要记录访问状态，避免重复搜索。

#### 2. 双向 BFS

- 同时从阿尔伯特和边界位置进行搜索。
- 在两个搜索队列相遇时停止，减少搜索空间。

---

### 总结

- **BFS 解法**直观易实现，适合处理滑行类问题。
- 时间复杂度为 **`O(side_length^3)`**，适合中等规模的网格。
- **DFS 和双向 BFS** 提供了不同的实现方式，适用于不同场景。