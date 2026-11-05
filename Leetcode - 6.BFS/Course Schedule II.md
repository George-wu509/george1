
### LintCode 616: 课程表 II（Course Schedule II）

在 `LintCode 616` 中，我们需要找到一种课程安排顺序，使得所有课程都能被完成。课程间存在依赖关系，如果无法完成所有课程，则返回空数组。

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

![[Gemini_Generated_Image_20tnpy20tnpy20tn.jpg]]
##### 一、 核心概念解析

1. **入度（In-degree）：**
    - 表示指向該節點的邊的數量。在課程問題中，**入度等於這門課的先修課總數**。
    - 入度為 `0` 表示該課程**沒有任何先修條件**，現在就可以直接修。  
    -    初始化數據結構 graph = {0:1,2, 1:3, 2:3}
		indegree = [0, 1, 1, 2]
		找到所有起點 queue = [ 0 ], result = []
		
2. **鄰接表（Adjacency List / `graph`）：**
    - 用來記錄課程之間的依賴方向。如果修完 `b` 才能修 `a`（即 `[a, b]`），則建立一條從 `b` 指向 `a` 的有向邊（`b -> a`）。
3. **雙端佇列（`deque`）：**
    - 用於暫存當前所有**入度為 0**（隨時可以修）的課程。

##### 二、 程式碼逐行詳細拆解
```python
from collections import defaultdict, deque

def find_order(num_courses, prerequisites):
    # 1. 建圖和入度陣列
    graph = defaultdict(list)
    indegree = [0] * num_courses

    for a, b in prerequisites:
        graph[b].append(a)  # 修完 b 才能修 a，所以邊的方向是 b -> a
        indegree[a] += 1  # a 需要先修課，所以 a 的入度 +1
```
- **目的：** 初始化數據結構。
- **邏輯：** 遍歷 `prerequisites` 陣列，對於每一對 `[a, b]`，代表 `b -> a`：
    - 在 `graph[b]` 中添加 `a`，表示「修完 `b` 後可以解鎖 `a`」。
    - `indegree[a] += 1`，表示「`a` 的先修課數量加 1」。

```Python
    # 2. 初始化佇列
    queue = deque([i for i in range(num_courses) if indegree[i] == 0])
    result = []
```
- **目的：** 找到所有**起點**。
- **邏輯：** 掃描所有課程，把初始入度為 `0`（完全不需要先修課）的課程全部放入佇列 `queue` 中。`result` 用來儲存最終修課的順序。
    
```Python
    # 3. BFS 遍歷
    while queue:
        course = queue.popleft()  # 從佇列取出一部當前可修的課程
        result.append(course)  # 將該課程放入修課順序清單中

        # 遍歷當前課程的鄰居（即依賴於當前課程的下一門課）
        for neighbor in graph[course]:
            indegree[neighbor] -= 1  # 因為修完了 course，neighbor 的先修課少了一門
            if indegree[neighbor] == 0:  # 如果 neighbor 的先修課都修完了
                queue.append(neighbor)  # 將 neighbor 加入佇列，準備修讀
```
- **目的：** 按照依賴順序依序修課並解除後續課程的限制。
- **邏輯：**
    1. 每次從佇列最前面彈出一門課程 `course`。
    2. 將 `course` 加入修課結果 `result`。
    3. 檢查所有需要先修 `course` 的後續課程 `neighbor`：
        - 將 `neighbor` 的入度減 1（相當於完成了一門先修課）。
        - 如果 `indegree[neighbor]` 降到了 `0`，說明這門後續課程的所有先修課都已修完，隨即將其加入佇列中。
            
```Python
    # 4. 檢查是否完成所有課程
    return result if len(result) == num_courses else []
```
- **目的：** 處理**環形依賴（Cycle）** 的情況。
- **邏輯：**
    - 如果圖中存在**死循環 / 環路**（例如 `A` 依賴 `B`，`B` 依賴 `A`），則環上的所有節點入度永遠不會降為 `0`，也就永遠無法加入佇列。
    - 因此，如果修完的課程總數 `len(result)` 等於 `num_courses`，說明成功排課，返回 `result`；否則說明存在環路無法完成所有課程，返回空陣列 `[]`。

### 三、 範例實導演練

假設：`num_courses = 4`, `prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]`

1. **建圖與計算入度：**
    
    - 依賴關係：`0 -> 1`, `0 -> 2`, `1 -> 3`, `2 -> 3`
        
    - `indegree` = `[0, 1, 1, 2]`（課程 0 入度為 0；課程 1、2 入度為 1；課程 3 入度為 2）
        
2. **初始化佇列：**
    
    - `queue` = `[0]`（只有課程 0 入度為 0）
        
    - `result` = `[]`
        
3. **BFS 過程：**
    
    - **Step 1:** 彈出 `0`，`result` = `[0]`
        
        - 減小 0 的鄰居（1 和 2）的入度：`indegree[1]` 變為 0，`indegree[2]` 變為 0。
            
        - 將 1 和 2 加入佇列 -> `queue` = `[1, 2]`
            
    - **Step 2:** 彈出 `1`，`result` = `[0, 1]`
        
        - 減小 1 的鄰居（3）的入度：`indegree[3]` 變為 1。
            
        - `queue` = `[2]`
            
    - **Step 3:** 彈出 `2`，`result` = `[0, 1, 2]`
        
        - 減小 2 的鄰居（3）的入度：`indegree[3]` 變為 0。
            
        - 將 3 加入佇列 -> `queue` = `[3]`
            
    - **Step 4:** 彈出 `3`，`result` = `[0, 1, 2, 3]`
        
        - 無後續鄰居，`queue` = `[]`
            
4. **最終判斷：**
    
    - `len(result) == 4`（等於 `num_courses`），返回 `[0, 1, 2, 3]`（或 `[0, 2, 1, 3]` 亦為有效解答）。






---

### 时间和空间复杂度分析

#### 时间复杂度
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