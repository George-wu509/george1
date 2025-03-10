
LintCode 611「骑士的最短路线」问题中，我们需要找到骑士（国际象棋中的马）从起点到目标点的最短移动步数。若目标点不可达，则返回 `-1`。

以下是该问题的中文详细解法以及具体示例。

---

## **LintCode 611: 骑士的最短路线 (Knight Shortest Path)**

### **题目描述**

在一个 `n x m` 的棋盘上，一个骑士（象棋中的马）从 **起点 `start`** 需要到达 **终点 `end`**，每次移动必须遵循 **国际象棋中的马走日规则**，即：

- `(x + 2, y + 1)`
- `(x + 2, y - 1)`
- `(x - 2, y + 1)`
- `(x - 2, y - 1)`
- `(x + 1, y + 2)`
- `(x + 1, y - 2)`
- `(x - 1, y + 2)`
- `(x - 1, y - 2)`

请找到 **从 `start` 到 `end` 的最短步数**。如果无法到达，则返回 `-1`。

---

Example: 样例 **例1:**

```python
输入:
[[0,0,0],
 [0,0,0],
 [0,0,0]]
source = [2, 0] destination = [2, 2] 
输出: 2
解释:
[2,0]->[0,1]->[2,2]
```

**例2:**

```python
输入:
[[0,1,0],
 [0,0,1],
 [0,0,0]]
source = [2, 0] destination = [2, 2] 
输出:-1
```


---

## **三种解法**

本题适用于 **DFS 递归、DFS 非递归（显式栈）、BFS** 解决，但由于要求**最短路径**，BFS 是最优解法。

---

### **方法 1: DFS 递归**

**解题思路**

- DFS 适用于路径搜索，但**无法保证最短路径**，因此通常不会用 DFS 来解最短路径问题。
- 在递归过程中，我们使用 `visited` 记录已访问的位置，防止重复搜索。
- 递归深度过大时，可能会导致 **栈溢出**。

### **步骤**

1. **初始化**：构建 `visited` 集合，避免重复搜索。
2. **递归搜索**：
    - 如果当前位置等于 `end`，更新 `min_steps` 记录最小步数。
    - 遍历 8 种骑士的移动方式，递归搜索。
    - 记录当前路径长度，进行回溯。
3. **返回 `min_steps`，如果不可达，返回 `-1`。**

### **代码**
```python
class Solution:
    def shortestPath(self, grid, source, destination):
        n, m = len(grid), len(grid[0])
        directions = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        self.min_steps = float('inf')

        def dfs(x, y, steps, visited):
            if (x, y) == destination:
                self.min_steps = min(self.min_steps, steps)
                return

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in visited and grid[nx][ny] == 0:
                    visited.add((nx, ny))
                    dfs(nx, ny, steps + 1, visited)
                    visited.remove((nx, ny))  # 回溯

        visited = set([(source[0], source[1])])
        dfs(source[0], source[1], 0, visited)
        
        return self.min_steps if self.min_steps != float('inf') else -1

```

### **时间复杂度**

- **最坏情况**：每个位置最多展开 8 个方向，总共有 `O(8^d)`，其中 `d` 是递归深度。
- **剪枝后**：复杂度较小，但仍然比 BFS 差。
- **空间复杂度**：`O(N*M)`（递归调用栈）。

---

### **方法 2: DFS 非递归（显式栈）**

**解题思路**

- 由于递归 DFS 有栈溢出风险，改用 **显式栈** 进行深度优先搜索。
- 但 **DFS 不能保证最短路径**，需要遍历所有可能路径。

### **步骤**

1. **初始化**：
    - 使用 `stack` 作为 DFS 栈，存放 `(当前位置, 当前步数)`。
    - 记录 `visited` 集合，避免重复访问。
2. **深度优先搜索 (使用 `stack`)**：
    - 取出栈顶元素，遍历所有可能移动。
    - 若抵达 `end`，更新最小步数。
    - 若合法，入栈。
3. **返回 `min_steps` 或 `-1`**。

### **代码**
```python
class Solution:
    def shortestPath(self, grid, source, destination):
        n, m = len(grid), len(grid[0])
        directions = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        stack = [(source[0], source[1], 0)]
        visited = set([(source[0], source[1])])
        min_steps = float('inf')

        while stack:
            x, y, steps = stack.pop()
            if (x, y) == destination:
                min_steps = min(min_steps, steps)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in visited and grid[nx][ny] == 0:
                    visited.add((nx, ny))
                    stack.append((nx, ny, steps + 1))

        return min_steps if min_steps != float('inf') else -1

```

### **时间复杂度**

- **最坏情况**：`O(8^d)`，其中 `d` 是搜索深度。
- **剪枝后**：仍然比 BFS 差。
- **空间复杂度**：`O(N*M)`（用于 `visited` 和 `stack`）。

---

### **方法 3: BFS（最优解）**

**解题思路**

- **BFS 是求最短路径的最佳方法**。
- BFS 按 **层遍历**，每次扩展所有可能的下一步，**第一个到达 `end` 的路径就是最短路径**。

### **步骤**

1. **初始化 `queue`**，存放 `(当前位置, 当前步数)`。
2. **BFS 遍历**：
    - 每次取出队首元素，遍历所有可能的下一步。
    - 若找到 `end`，直接返回步数（因为 BFS 先到达的路径一定是最短的）。
    - 若新位置未访问，加入 `queue` 并标记 `visited`。
3. **如果 BFS 结束仍未找到 `end`，返回 `-1`。**

### **代码**
```python
from collections import deque

class Solution:
    def shortestPath(self, grid, source, destination):
        n, m = len(grid), len(grid[0])
        directions = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        queue = deque([(source[0], source[1], 0)])
        visited = set([(source[0], source[1])])

        while queue:
            x, y, steps = queue.popleft()
            if (x, y) == destination:
                return steps

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in visited and grid[nx][ny] == 0:
                    visited.add((nx, ny))
                    queue.append((nx, ny, steps + 1))

        return -1

```

### **时间复杂度**

- **BFS 只遍历 `N*M` 个格子，复杂度为 `O(N*M)`**。
- **空间复杂度**: `O(N*M)`（用于 `queue` 和 `visited`）。

---

## **总结**

|方法|适用情况|时间复杂度|额外空间|
|---|---|---|---|
|**DFS 递归**|适用于搜索路径，但可能超时|`O(8^d)`|`O(N*M)`|
|**DFS 非递归 (栈)**|适用于路径搜索，但不能保证最短路径|`O(8^d)`|`O(N*M)`|
|**BFS (最优解)**|适用于最短路径搜索，保证最优解|`O(N*M)`|`O(N*M)`|

**推荐使用：BFS**，因为它保证能最快找到最短路径。