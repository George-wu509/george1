
![[Pasted image 20240914170851.png]]
![[Pasted image 20240914170923.png]]

1. BFS vs DFS 複雜度
    
    時間複雜度= O(V + E), V頂點個數, E邊數
    
    BFS空間複雜度取決於寬度
    
    DFS空間複雜度取決於深度
    

5 DFS時間複雜度通用公式 = O(方案總數 * 構造每個方案的時間)

6 DFS都可以用遞歸(Recursion) 實現. 遞歸三步驟: 1遞歸定義2 遞歸拆解3 遞歸出口

7 絕大多數二叉樹(Binary Tree)的問題都可以用DFS求解. 遇到找所有方案的題基本上就是用DFS

8 90% DFS的題都是二叉樹, 10%DFS的題是組合(combination) 或排列(permutation), [1,2,3]和[3,2,1]是同一個組合但不同的排列

![[Pasted image 20240914171001.png]]

## **Lintcode模板 - 深度优先搜索 DFS**

## **使用条件**

- 找满足某个条件的所有方案（99%）
- 二叉树 `Binary Tree` 的问题（90%）
- 组合问题（95%）
    - 问题模型：求出所有满足条件的“组合”
    - 判断条件：组合中的元素是顺序无关的
- 排列问题（95%）
    - 问题模型：求出所有满足条件的“排列”
    - 判断条件：组合中的元素是顺序“相关”的

## **不要用 `DFS` 的场景**

- 连通块问题（一定要用 `BFS`，否则 `StackOverflow`）
- 拓扑排序（一定要用 `BFS`，否则 `StackOverflow`）
- 一切 `BFS` 可以解决的问题

## **复杂度**

- 时间复杂度：_O_(方案个数∗构造每个方案的时间)
    
    O(方案个数 * 构造每个方案的时间)
    
    - 树的遍历 ：_O_(_n_)
    - 排列问题 ：_O_(_n_!∗_n_)
    - 组合问题 ：_O_(2_n_∗_n_)

**代码模板**
![[Pasted image 20240914171047.png]]

## **三种解法**

本题适用于 **DFS 递归、DFS 非递归（显式栈）、BFS** 解决，但由于要求**最短路径**，BFS 是最优解法。
解釋:
```python
# ----- DFS 递归 -----

   def dfs([x, y]):
      for dx, dy in directions:	
         dfs(x+dx, y+dy)
   def([x0, y0])


# ----- DFS 非递归 -----

   stack = [(x0,y0)] 
   while stack:
      x,y = stack.pop()
      for dx, dy in directions:
	      stack.append(x+dx, y+dy)
   
# ----- BFS -----
   from collections import deque
   
   queue = deque([x0,y0])      
   while queue:
      x,y = queue.popleft()     
      for dx, dy in directions:
	      queue.append(x+dx, y+dy)

```

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



| **题目编号**           | **题目名称 (英文/中文)**                                               | **题目简述 (中文)**                      | **样例**                                                                                                                                                  | **解法**                              |
| ------------------ | -------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| 15<br>**<br>(m)    | 全排列 <br>[[Permutations]]                                       | 给定一个没有重复数字的数组，返回所有可能的排列。           | 输入：<br>[1,2,3]<br>输出：[<br>  [1,2,3],<br>  [1,3,2],<br>  [2,1,3],<br>  [2,3,1],<br>  [3,1,2],<br>  [3,2,1] ]                                             | 使用 DFS 和回溯生成所有排列组合。                 |
| 816<br>*<br>(h)    | 旅行商问题 <br>[[Traveling Salesman Problem]]<br>                   | 给定城市的距离矩阵，找到访问所有城市的最短路径并返回起点。      | 输入: <br>n = 3<br>tuple = [<br>[1,2,1],<br>[2,3,2],<br>[1,3,3]]<br>输出: 3<br>                                                                             | 使用 DFS + 剪枝搜索所有路径，记录最小路径和。          |
| 17<br>**<br>(m)    | 子集 <br>[[Subsets]]                                             | 给定一个没有重复元素的数组，返回所有可能的子集。           | 輸入<br>[1,2,3]<br>輸出<br>[ [], [1], [1, 2],<br> [1, 2, 3], [1, 3], <br>[2], [2, 3], [3] ]                                                                 | 使用 DFS 生成所有可能的组合，逐步扩展路径。            |
| 18<br>*<br>(m)     | 子集II <br>[[SubsetsII]]                                         | 给定一个可能包含重复元素的数组，返回所有可能的子集，且子集不能重复。 | 输入：<br>nums = [1,2,2] <br>输出：[ <br>  [2], [1], [1,2,2], <br>  [2,2], [1,2],[] ]                                                                         | 对数组排序后使用 DFS，跳过重复元素生成子集。            |
| 152<br><br>(m)     | [[Combinations]]组合                                             | 找到从 n 个数字中选择 k 个数字的所有组合。           | 输入:  <br>n = 4  <br>k = 2  <br>输出:  <br>[ [2,4],[3,4],[2,3],[1,2],[1,3],[1,4] ]                                                                         | 使用回溯法和哈希表记录每次选择的数字。                 |
| 153<br><br>(m)<br> | [[Combination Sum II]]组合II                                     | 找到数组中和等于目标值的所有不重复组合。               | 输入:  <br>candidates = [10,1,2,7,6,1,5]  <br>target = 8  <br>输出:  <br>[ [1,1,6],[1,2,5],[1,7],[2,6] ]                                                    | 使用回溯法结合哈希表记录已访问的数字避免重复。             |
| 427<br>*<br>(m)    | 生成括号 <br>[[Generate Parentheses]]<br>                          | 给定一个正整数 n，生成所有合法的括号组合。             | 输入: 2<br>输出: <br>["()()", <br>"(())"]                                                                                                                   | 使用 DFS 和回溯生成合法括号组合，动态维护左右括号数量。      |
| 582<br>*<br>(h)    | 单词拆分II <br>[[Word BreakII]]                                    | 给定一个字符串和一个单词字典，返回该字符串的所有可能分割方案。    | 输入：<br>"lintcode"，<br>["de","ding",<br>"co","code",<br>"lint"]<br>输出：<br>["lint code", <br>"lint co de"]                                                | 使用 DFS 搜索分割点，结合回溯生成所有可能方案。          |
| 132<br>*<br>(h)    | 单词搜索 II [[Word Search II]]                                     | 给定一个二维字符网格和单词字典，找出所有字典中的单词在网格中的位置。 | 输入:  <br>board = [["o","a","b","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]  <br>输出: ["oath","eat"] | 使用字典树存储单词，结合DFS遍历网格查找。              |
| 1848<br>*<br>(h)   | 单词搜索 III [[Word Search III]]                                   | 给定一个单词列表和一个二维字符网格，找出所有单词在网格中的位置。   | 输入:  <br>board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]  <br>输出: ["oath","eat"] | 使用字典树构建单词索引，结合DFS遍历网格查找所有单词。        |
| 1909<br>*<br>(m)   | 订单分配 <br>[[Order Allocation]]                                  | 给定一个数组，返回所有可能的排列组合，支持重复数字。         | 输入：[<br>[1,2,4],<br>[7,11,16],<br>[37,29,22] ]<br>输出：<br>[1,2,0]                                                                                        | 使用 DFS 和回溯生成所有排列，跳过重复组合。            |
| 1360<br>*<br>(m)   | 对称树 <br>[[Symmetric Tree]] <br>                                | 判断一棵二叉树是否为对称树。                     | 输入: <br>{1,2,2,3,4,4,3}<br>输出: true                                                                                                                     | 使用 DFS 递归比较左右子树是否对称，检查值和结构是否一致。     |
| 1271<br>*<br>(h)   | 查找集群内的<br>「关键连接」 <br>[[Critical Connections in a Network]]<br> | 找到网络中所有的关键连接，删除这些连接会导致网络分裂。        | 輸入: 4<br>[ [0,1],[1,2],<br>[2,0],[1,3] ]<br>輸出<br>[ [1,3] ]                                                                                             | 使用 Tarjan 算法实现 DFS，记录时间戳和低链接值，找出桥边。 |
| 802<br>*<br>(h)    | 数独 <br>[[Sudoku Solver]]<br>                                   | 求解一个数独问题，返回其唯一解。                   |                                                                                                                                                         | 使用 DFS 尝试填充每个空格，验证数独规则，找到唯一解法。      |







