
lintcode 803

### LeetCode 317: Shortest Distance from All Buildings

#### **问题描述**

给定一个二维网格，其中：

- `0` 表示空地，
- `1` 表示建筑物，
- `2` 表示障碍物。

你需要找到一块空地，使得从该空地到所有建筑物的总距离最短，并返回这个最短距离。如果无法找到这样的空地，返回 `-1`。

```python
grid = 
[ [1, 0, 2, 0, 1],
  [0, 0, 2, 0, 0],
  [0, 0, 0, 0, 0]
]

```

---

### **解法：使用 BFS**

#### **核心思路**

1. **使用 BFS 遍历每个建筑物：**
    
    - 从每个建筑物出发，计算到所有空地的距离。
    - 用两个辅助矩阵记录：
        - `distances`：记录空地累积的距离。
        - `reach_count`：记录空地可以到达的建筑物数量。
2. **遍历所有网格空地：**
    
    - 找到 `reach_count` 等于建筑物总数的空地。
    - 返回这些空地中 `distances` 的最小值。

---

#### **辅助数据结构**

1. **`distances` 矩阵：**  
    记录从所有建筑物出发到达每个空地的累积距离。
    
2. **`reach_count` 矩阵：**  
    记录每个空地可以到达的建筑物数量。
    

---

### **逐步解析**

#### **初始化**

**输入网格：**
```css
grid = 
[  [1, 0, 2, 0, 1],
  [0, 0, 2, 0, 0],
  [0, 0, 0, 0, 0]
]

```

- **建筑物总数：** `building_count = 2`
- 初始化两个辅助矩阵：
    
```csharp
distances = [
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]
]

reach_count = [
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]
]

```
    

---

#### **BFS 遍历每个建筑物**

##### **从 (0, 0) 出发：**

- **初始化 BFS 队列：**  
    `queue = deque([(0, 0, 0)])`，第三个元素表示距离。
- 遍历过程：
    1. 取出 `(0, 0, 0)`，向四个方向扩展：
        - `(1, 0)` 是空地，加入队列，距离为 `1`：  
            更新 `distances` 和 `reach_count`：
            
```csharp
distances = [
  [0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]
]

reach_count = [
  [0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]
]

```
            
        - 其他方向无效（越界或障碍物）。
    2. 取出 `(1, 0, 1)`，继续扩展：
        - `(2, 0)` 是空地，加入队列，距离为 `2`。

##### **从 (0, 4) 出发：**

- 重复类似过程，从 (0, 4) 的 BFS 开始更新 `distances` 和 `reach_count`。

---

#### **查找最优空地**

- 遍历所有空地 `(i, j)`：
    - 如果 `reach_count[i][j]` 等于 `building_count`，将其距离加入候选最小值。
- 最后返回最小值。

---
Example:
例如，给定三个建筑物`(0,0)`,`(0,4)`,`(2,2)`和障碍物`(0,2)`:
```
	1 - 0 - 2 - 0 - 1
	|   |   |   |   |
	0 - 0 - 0 - 0 - 0
	|   |   |   |   |
	0 - 0 - 1 - 0 - 0
```
点`(1,2)`是建造房屋理想的空地，因为`3+3+1=7`的总行程距离最小。所以返回`7`。


```python
from collections import deque

def shortest_distance(grid):
    rows, cols = len(grid), len(grid[0])
    building_count = sum(cell == 1 for row in grid for cell in row)

    # 辅助矩阵
    distances = [[0] * cols for _ in range(rows)]
    reach_count = [[0] * cols for _ in range(rows)]

    def bfs(start_row, start_col):
        visited = [[False] * cols for _ in range(rows)]
        queue = deque([(start_row, start_col, 0)])  # (行, 列, 距离)
        visited[start_row][start_col] = True

        while queue:
            row, col, dist = queue.popleft()

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols and not visited[new_row][new_col]:
                    if grid[new_row][new_col] == 0:  # 仅处理空地
                        visited[new_row][new_col] = True
                        queue.append((new_row, new_col, dist + 1))
                        distances[new_row][new_col] += dist + 1
                        reach_count[new_row][new_col] += 1

    # 对每个建筑物执行 BFS
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                bfs(r, c)

    # 查找最短距离
    min_distance = float('inf')
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and reach_count[r][c] == building_count:
                min_distance = min(min_distance, distances[r][c])

    return min_distance if min_distance != float('inf') else -1

# 测试
grid = [
  [1, 0, 2, 0, 1],
  [0, 0, 2, 0, 0],
  [0, 0, 0, 0, 0]
]
print(shortestDistance(grid))  # 输出: 7

```
pass


### **逐步执行过程**

#### 初始化：

```csharp
distances = [
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]
]

reach_count = [
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]
]

```

---

#### 从 (0, 0) 出发 BFS 更新：

- 经过 `(1, 0)`、`(2, 0)` 等空地，更新 `distances` 和 `reach_count`。

#### 从 (0, 4) 出发 BFS 更新：

- 经过 `(1, 4)`、`(2, 4)` 等空地，继续更新矩阵。

---

#### 最终矩阵：

```csharp
distances = [
  [0, 0, 0, 0, 0],
  [1, 2, 0, 2, 1],
  [2, 4, 0, 4, 2]
]

reach_count = [
  [0, 0, 0, 0, 0],
  [1, 2, 0, 2, 1],
  [2, 2, 0, 2, 2]
]

```

- 找到 `reach_count == building_count` 的位置，取 `distances` 最小值：`7`。

---

### **时间与空间复杂度**

1. **时间复杂度：**
    
    - 每个建筑物进行一次 BFS，复杂度为 O(m×n)O(m \times n)O(m×n)。
    - 总复杂度为 O(b×m×n)O(b \times m \times n)O(b×m×n)，其中 bbb 是建筑物数量。
2. **空间复杂度：**
    
    - BFS 使用队列，空间复杂度为 O(m×n)O(m \times n)O(m×n)。

---

### **总结**

- 使用 BFS 能高效计算每个建筑物到空地的最短距离。
- 辅助矩阵 `distances` 和 `reach_count` 记录累积状态，避免重复计算。
- 适用于类似最短路径、多源 BFS 的问题。