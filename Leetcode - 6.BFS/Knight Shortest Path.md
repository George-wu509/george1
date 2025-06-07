
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


最佳解 BFS
```python
from collections import deque
DIRECTIONS = [(1,-2),(1,2),(-1,-2),(-1,2),(2,1),(2,-1),(-2,1),(-2,-1)]

class Solution:
    def shortpath(self, grid, source, destination):
        if not grid or not grid[0]:
            return
        queue = deque([(source.x, source.y, 0)])
        visited = set([(source.x, source.y)])
        
        while queue:
            x, y, steps = queue.popleft()
            if (x, y) == (destination.x, destination.y):
                return steps
            
            for dx,dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny, steps + 1))
                    
        return -1
```
pass


它涉及到 Python 中 `set` 的**初始化（構造函數）**和**添加元素（`add` 方法）**之間的細微但重要的區別。理解這個區別對於正確使用集合非常重要。

### `set` 的初始化（構造函數）`set([...])`

在 Python 中，當您使用 `set()` 構造函數來初始化一個集合時，例如 `set(iterable)`，這個構造函數期望一個**可迭代對象 (iterable)** 作為參數。它會遍歷這個可迭代對象中的**每一個元素**，並將這些元素逐個添加到新的集合中。

舉例來說：

- `set([1, 2, 3])`：這裡 `[1, 2, 3]` 是一個列表（可迭代對象），它有三個元素 `1`、`2`、`3`。所以，集合會被初始化為 `{1, 2, 3}`。
- `set((1, 2, 3))`：這裡 `(1, 2, 3)` 是一個元組（可迭代對象），它也有三個元素 `1`、`2`、`3`。集合同樣會被初始化為 `{1, 2, 3}`。

現在來看您的例子 `visited = set([(source[0], source[1])])`：

1. `[(source[0], source[1])]` 是一個**列表**。
2. 這個列表只包含**一個元素**，就是元組 `(source[0], source[1])`。
3. 當 `set()` 構造函數接收這個列表時，它會遍歷這個列表，並將列表中**唯一的那個元素** `(source[0], source[1])` 添加到新的集合中。
4. 所以，`visited` 集合會被正確地初始化為 `{ (source[0], source[1]) }`，其中 `(source[0], source[1])` 作為一個完整的座標點被儲存。

**錯誤的例子**是 `set((source[0], source[1]))`： 如果 `source = [0, 0]`，那麼 `(source[0], source[1])` 就是 `(0, 0)`。 當您寫 `set((0, 0))` 時，`set()` 會將元組 `(0, 0)` 視為一個可迭代對象，並遍歷它的**內部元素**。它會先取出第一個 `0` 加到集合，再取出第二個 `0` 加到集合。由於集合不儲存重複元素，最終結果會是 `{0}`。這顯然不是我們想要儲存 `(0, 0)` 這個座標點的目的。

### `set` 的添加元素方法 `add()`

當您使用 `set.add(element)` 方法向一個**已存在的集合**添加元素時，這個方法只接受**一個單一的參數**，這個參數就是您想要添加到集合中的**那個元素本身**。這個元素必須是**可雜湊 (hashable)** 的。

舉例來說：

- 如果 `my_set = {1, 2}`，然後 `my_set.add(3)`，結果是 `{1, 2, 3}`。
- 如果 `my_set = { (1, 2) }`，然後 `my_set.add((3, 4))`，結果是 `{ (1, 2), (3, 4) }`。這裡 `(3, 4)` 是一個元組，元組是可雜湊的。

現在來看您的例子 `visited.add((nx, ny))`：

1. `(nx, ny)` 是一個**元組**，它代表一個新的座標點。
2. 元組是 Python 中**可雜湊**的數據類型。
3. `visited.add()` 方法直接將這個完整的元組 `(nx, ny)` 作為一個單一的元素添加到 `visited` 集合中。
4. 所以，`visited` 集合會正確地包含所有已訪問的座標點元組。

**錯誤的例子**是 `visited.add([(nx, ny)])`：

1. `[(nx, ny)]` 是一個**列表**。
2. 列表在 Python 中是**不可雜湊**的（因為列表是可變的，其內容可以在創建後改變，這違反了雜湊的要求）。
3. 當您嘗試將一個不可雜湊的列表添加到集合時，會引發 `TypeError: unhashable type: 'list'` 錯誤。

### 總結

簡而言之，關鍵在於：

- **`set()` 構造函數**：它接收一個**可迭代對象**，並將該可迭代對象中的**每個獨立元素**添加到集合中。如果您想讓一個元組作為單一元素被添加，您需要提供一個只包含這個元組的可迭代對象（例如 `[(source[0], source[1])]`）。
- **`add()` 方法**：它接收一個**單一的、可雜湊的元素**，並將這個元素直接添加到集合中。所以您需要直接傳入 `(nx, ny)` 這個元組，而不是將它包裹在一個列表中。

理解這些基本概念能幫助您避免在使用集合時常見的錯誤，並更高效地解決問題。





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