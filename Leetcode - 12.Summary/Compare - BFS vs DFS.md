
|                                       |     |
| ------------------------------------- | --- |
| [[#### BFS vs DFS pseudo code]]       |     |
| [[#### BFS vs DFS何時使用]]               |     |
| [[#### 例題(島嶼的個數, permutation)完整Code]] |     |
| [[#### BFS 使用 Queue與 DFS 使用 Stack]]   |     |
|                                       |     |
#### BFS vs DFS pseudo code
```
BFS vs DFS pseudo code
```

BFS 廣度優先搜索(Breadth-First Search)
```python
visited = set()
for i
for j
   if grid[i][j] not in visited:
      BFS(grid, i, j, visited)

BFS(grid, x, y, visited):
   qu = deque([x,y])
   visited.add((x,y))
   while qu:
      x,y = queue.popleft()
      for delta_x, delta_y in DIRECTIONS:
         next_x = x + delta_x
         next_y = y + delta_y
            
         queue.append((next_x, next_y))
         visited.add((next_x, next_y))
```
ref: [[Number of Islands]]
explain:
step1: Create visited=set(), DIRECTIONS
step2: 先遍歷Array(2D array)在每個沒有visited點執行BFS()
step3: 在BFS裡面(沒有遞迴), 先create deque([x,y]), 並標示visited
step4: 接著 while queue, 並把這個點(x,y)從queue左邊移除
step5: 用for迴圈, deque加入這個點周圍(DIRECTIONS)(x+-dx,y+-y)
step6: 在for迴圈裏面也把這些點(x+-dx,y+-y)標示成visited.
step7: 接著 while queue=(x+-dx,y+-y), 也依序移除這些點
step8. 並在每個這些點都用for迴圈加入周圍的點到deque以及visited

![[Pasted image 20261101105315.png]]
![[Pasted image 20261101105502.png|401]]

DFS 深度優先搜索(Depth-First Search)
```python
results = []
visited = [False]*len(nums)

DFS(nums, visited, num, results):
   到tree底則append加入results
   
   for i in nums全部
      visited[i] = True
      nums.append(num[i])
      DFS()
      nums.pop()
```
ref: [[Permutations]]
Explain:
step1: Create visited=[False]* len(nums)
step2: 在DFS(nums, visited, num, results)裡面(有遞迴)
step3: DFS()開頭如果走到底則num=[1,2,3]加入results成= [ [1,2,3], ]
step4: 在DFS()之後用for迴圈i in nums, 如果nums[i]沒有在visited, 則加
      入到visited裡面
step5: 則加入到visited裡面 -> num用append加入nums[i] -> 遞迴DFS           -> num用pop()移除 -> 從visited移除nums[i] 
```python
    visited[i] = True         #i加入到visited
    num.append(nums[i])       #數值 append加入到num
    self.dfs(nums, visited, num, results)
    num.pop()                 #數值 移除 pop from num
    visited[i] = False        #從visited移除i
```


![[Pasted image 20261101110143.png|397]]

![[Pasted image 20261101111953.png|382]]




#### BFS vs DFS何時使用
```
BFS vs DFS何時使用
```

能用BFS (沒有遞迴)的一定不要用DFS(有遞迴)（除非面试官特别要求）

![[Pasted image 20261101115147.png]]

![[Pasted image 20261101115211.png]]



#### 例題(島嶼的個數, permutation)完整Code
```
例題(島嶼的個數, permutation)完整Code
```

|                  |                             |                     |                                                                                                                        |                            |
| ---------------- | --------------------------: | ------------------: | ---------------------------------------------------------------------------------------------------------------------: | -------------------------: |
| 433<br>**<br>(e) | 岛屿的个数 [[Number of Islands]] | 给定一个二维网格，计算其中岛屿的数量。 | 输入：<br>[<br>  [1,1,0,0,0],<br>  [0,1,0,0,1],<br>  [0,0,0,1,1],<br>  [0,0,0,0,0],<br>  [0,0,0,0,1]<br>]<br><br>输出：<br>3 | 使用 BFS 遍历每个岛屿，将相邻陆地标记为已访问。 |
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


|                           |                          |                          |                                                                                                                 |                     |
| ------------------------- | -----------------------: | -----------------------: | --------------------------------------------------------------------------------------------------------------: | ------------------: |
| 15<br>**<br>(m)<br><br>PR | 全排列 <br>[[Permutations]] | 给定一个没有重复数字的数组，返回所有可能的排列。 | 输入：<br>[1,2,3]<br><br>输出：[<br>  [1,2,3],<br>  [1,3,2],<br>  [2,1,3],<br>  [2,3,1],<br>  [3,1,2],<br>  [3,2,1] ] | 使用 DFS 和回溯生成所有排列组合。 |
```python
class Solution:
    def permute(self, nums):
        results = []
        visited = [False]*len(nums)
        self.dfs(nums, visited, [], results)
        return results
        
    def dfs(self, nums, visited, num, results):
        if len(num) == len(nums):
            results.append(list(num))
            return
        
        for i in range(len(nums)):
            if not visited[i]:
                visited[i] = True
                num.append(nums[i])
                self.dfs(nums, visited, num, results)
                num.pop()
                visited[i] = False
                
sol = Solution()
nums = [1,2,3]
result = sol.permute(nums)
print(result)
```



#### BFS 使用 Queue與 DFS 使用 Stack
```
在leetcode的有些題目有時候適用用BFS有時候用DFS解, 納為什麼通常BFS會使用deque, 而DFS用stack的資料結構? 這是一定的嗎?還是看題目有時候用stack有時候queue? 是因為通常DFS需要後進先出而BFS先進先出?
```
選擇 **BFS 使用 Queue（FIFO）** 與 **DFS 使用 Stack（LIFO）** 並不是隨意的約定，而是由這兩種搜尋演算法的核心邏輯所決定的。

### 1. 為什麼 BFS 用 Queue（先進先出 FIFO）？

BFS（廣度優先搜尋）的精神是**逐層擴展（Level-order Traversal）**。

- **核心需求**：你必須先處理完「距離起點最近」的所有節點，才能處理下一層的節點。
    
- **FIFO 的運作**：
    
    - 先被發現的節點（第 $K$ 層）會先從 Queue 的前端被取出處理。
        
    - 處理時產生的新節點（第 $K+1$ 層）會被放入 Queue 的後端排隊。
        
    - 這種機制確保了 Queue 裡面的節點永遠保持**階層順序**，保證找到目標時走過的是**最短路徑**。
        

> **為什麼 Python 要用 `collections.deque` 而不是普通 `list`？**
> 
> Python 的 `list.pop(0)` 時間複雜度是 $O(N)$（因為後面所有元素都要往前挪），而 `deque.popleft()` 是 $O(1)$，因此做 BFS 必須用 `deque`。

### 2. 為什麼 DFS 用 Stack（後進先出 LIFO）？

DFS（深度優先搜尋）的精神是**一條路走到底（Go Deep First）**，碰到死胡同再回溯（Backtrack）。

- **核心需求**：你剛發現的新節點（最新的子節點），必須**立刻成為下一個探索的目標**。
    
- **LIFO 的運作**：
    
    - 最新被壓入（Push）Stack 的節點會最先被彈出（Pop）處理。
        
    - 這讓搜尋路徑能持續沿著單一分支向下延伸，直到無法再走時，才退回上一個分支點（Backtracking）。
        

### 3. 這是一定的嗎？還是可以互換？

技術上**資料結構決定了搜尋行為**。如果你把 BFS 的 Queue 改成 Stack，它就會變成迭代版的 DFS；反之亦然。

但在實際寫 LeetCode 時，有以下幾點變化與例外：

#### ① DFS 通常直接用「系統呼叫棧（Call Stack）」，不手動宣告 Stack

大多數 DFS 題目都會寫成**遞迴（Recursion）**。呼叫函式本身就會利用系統底層的 Call Stack，所以程式碼裡不需要寫 `stack = []`。只有在要避免遞迴過深導致 Stack Overflow，或是要求迭代（Iterative）解法時，才會手動維護一個 Stack。

#### ② 特殊題目：優先權佇列 PriorityQueue / Min-Heap

當圖的邊有權重（例如 Weighted Shortest Path）時，單純的 FIFO Queue 無法保證最短路徑。這時 BFS 會升級成 **Dijkstra 演算法**，使用的資料結構會從 Queue 變成 **PriorityQueue（Min-Heap）**。

#### ③ 特殊題目：雙端佇列 Deque 用於 0-1 BFS

在邊的權重只有 0 和 1 的格狀圖或地圖題中，我們會使用 `deque`：

- 權重為 0 的邊走到的新節點：放在 `deque.appendleft()`（優先處理）。
    
- 權重為 1 的邊走到的新節點：放在 `deque.append()`（正常排隊）。
    

### 總結對比

|**演算法**|**核心目標**|**資料結構**|**運作特性**|**經典應用場景**|
|---|---|---|---|---|
|**BFS**|按距離/階層橫向擴展|**Queue / Deque**|先進先出 (FIFO)|無權重圖的最短路徑、層序遍歷 (Level Order)|
|**DFS**|沿著單一路徑縱向深入|**Stack** (或遞迴 Call Stack)|後進先出 (LIFO)|全排列/組合搜尋、連通分量 (Islands)、拓撲排序|