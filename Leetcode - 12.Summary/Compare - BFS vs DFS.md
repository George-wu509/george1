
|                                           |     |
| ----------------------------------------- | --- |
| [[#### BFS vs DFS pseudo code]]           |     |
| [[#### BFS vs DFS何時使用]]                   |     |
| [[#### 例題(島嶼的個數, permutation)完整Code]]     |     |
| [[#### BFS 使用 Queue與 DFS 使用 Stack]]       |     |
| [[#### 例子解釋BFS 使用 Queue與 DFS 使用 Stack]]   |     |
| [[#### 甚麼時候用BFS? DFS?]]                   |     |
| [[#### lintcode的老師建議用BFS解題目]]             |     |
| [[#### 遞迴爆棧跟DFS用Stack, BFS用deque跟Heap關係]] |     |
| [[#### 常用演算法與資料結構的記憶體分布]]                 |     |
|                                           |     |
|                                           |     |
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

BFS完整Code
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
|                           |                          |                          |                                                                                                                 |                     |
DFS完整Code
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
> Python 的 `list.pop(0)` 時間複雜度是 $O(N)$（因為後面所有元素都要往前挪），而 `deque.popleft()` 是 $O(1)$，因此做 BFS 必須用 `deque`。

### 2. 為什麼 DFS 用 Stack（後進先出 LIFO）？

DFS（深度優先搜尋）的精神是**一條路走到底（Go Deep First）**，碰到死胡同再回溯（Backtrack）
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




#### 例子解釋BFS 使用 Queue與 DFS 使用 Stack
```
幫我用tree的圖用具體的例子(島嶼的個數vs permutation)一步步解釋為什麼 BFS 用 Queue（先進先出 FIFO), 為什麼 DFS 用 Stack（後進先出 LIFO）
```
要理解 BFS 和 DFS 為什麼分別需要 Queue 和 Stack，我們可以直接從它們的「探索目的」出發：**BFS 像是在水面投下石頭的漣漪，需要「同心圓向外擴張」（先進先出）；而 DFS 像是在走迷宮，需要「一條路走到黑，撞牆才回頭」（後進先出）**。

我們用你提到的兩個經典問題，搭配樹狀圖與資料結構的變化來一步步拆解。

### 1. 為什麼 DFS 用 Stack (LIFO)？

**具體例子：排列組合 (Permutations) — 找出 `[A, B, C]` 的所有排列**

在排列組合中，我們的目標是盡快湊齊一個完整的組合（例如 `[A, B, C]`），然後**退回上一步**（Backtrack），換另一個選項（例如換成 `[A, C, B]`）。這就是典型的「深度優先」。

**決策樹 (Decision Tree)：**
```
               [ ]  <-- Root (第 0 層)
            /   |   \
          /     |     \
       [A]     [B]     [C]  <-- 第 1 層
       / \
      /   \
  [A,B]   [A,C]  ...  <-- 第 2 層
    |       |
 [A,B,C] [A,C,B]  ... <-- 第 3 層 (完整結果)
```

**Stack (後進先出) 運作步驟：** Stack 的特性是「最新加入的選項，最先被拿出來處理」，這強迫演算法不斷往深處走。

1. **初始狀態**：將 Root 放進 Stack。
    - Stack: `[ [] ]`
2. **第一步**：彈出 `[]`，將它的下一步可能 `[C]`, `[B]`, `[A]` 依序壓入 Stack。
    - Stack: `[ [C], [B], [A] ]` （`[A]` 在最頂端）
3. **第二步 (關鍵)**：因為是 LIFO，我們彈出最頂端的 `[A]`，然後將它後續的可能 `[A,C]`, `[A,B]` 壓入。
    - Stack: `[ [C], [B], [A,C], [A,B] ]`
    - _注意：我們沒有去管 `[B]` 或 `[C]`，而是直接往 `[A]` 的深處 `[A,B]` 鑽進去了！這就是 Stack 造成的深度優先效果。_
4. **第三步**：彈出 `[A,B]`，將 `[A,B,C]` 壓入。
    - Stack: `[ [C], [B], [A,C], [A,B,C] ]`
5. **第四步**：彈出 `[A,B,C]`，達到長度 3，輸出第一個排列！
6. **第五步 (回溯)**：現在 Stack 頂端是 `[A,C]`，我們自然而然地退回到了上一層的分支，繼續往 `[A,C,B]` 探索。

### 2. 為什麼 BFS 用 Queue (FIFO)？

**具體例子：島嶼的個數 (Number of Islands) — 找出相連的陸地**

在島嶼問題中，當我們踩到一塊陸地 `1` 時，我們需要把「與它直接相連」的陸地全部找出來，然後再找「與相連陸地相連」的更外圍陸地。這需要一層一層向外擴展，確保同一座島嶼被均勻地探索完畢。

**網格地圖與樹狀展開：** 假設地圖左上角有一座島嶼：
```
(0,0)=1  (0,1)=1  (0,2)=0
(1,0)=1  (1,1)=0  (1,2)=0
```

**BFS 探索樹：**
```
              (0,0)         <-- 發現新島嶼的起點 (第 1 圈)
              /   \
          (0,1)   (1,0)     <-- 與起點直接相鄰的陸地 (第 2 圈)
```

**Queue (先進先出) 運作步驟：** Queue 的特性是「先排隊的先處理」，這保證了演算法必須把「目前這一層（圈）」全部處理完，才能處理「下一層」。
1. **初始狀態**：發現陸地 `(0,0)`，加入 Queue，並標記為已拜訪。
    - Queue: `[ (0,0) ]`
2. **第一步**：取出 `(0,0)`。將它右邊 `(0,1)` 和下面 `(1,0)` 的相鄰陸地加入排隊。
    - Queue: `[ (0,1), (1,0) ]`
3. **第二步 (關鍵)**：因為是 FIFO，我們取出最先排隊的 `(0,1)`。它的周圍沒有新的未拜訪陸地。
    - Queue: `[ (1,0) ]`
    - _注意：我們接下來處理的是同屬於「第 2 圈」的 `(1,0)`，而不是因為發現了新節點就立刻衝過去。這就是 Queue 造成的廣度優先（擴張）效果。_
4. **第三步**：取出 `(1,0)`。周圍也沒有新陸地。
    - Queue: `[ ]` （空了）
5. **結束**：Queue 清空，代表這座島嶼的所有相連區塊已經被我們「一層層」地完全掃描完畢。

資料結構的特性直接決定了探索的行為模式。Stack 的「喜新厭舊」讓 DFS 能夠一路直闖到底，非常適合用來尋找路徑或窮舉組合；而 Queue 的「先來後到」讓 BFS 必須穩紮穩打地往外擴展，這讓它成為解決最短路徑和連通區塊（如島嶼）的最佳工具。




#### 甚麼時候用BFS? DFS?
```
Leetcode question 島嶼的個數, 騎士的最短路線等題目都會建議使用BFS method解題, 那都是像在一個2D map裡面有設定DIRECTIONS可能代表上下左右等移動. 那為什麼太平洋大西洋水流問題似乎是建議使用DFS method? 因為這問題看起來也很像是2D map然後往島內部移動. 所以是甚麼樣的題目會建議使用BFS(就我的理解應該可以用BFS解的應該可以用DFS解)? 甚麼會建議用DFS? 那像拓扑排序為何是使用BFS?
```
你的直覺對了一半：**如果題目只求「連通性（能不能走到、有幾塊區域）」，BFS 和 DFS 完全可以互換；但如果題目要求「無權重最短路徑（最少幾步、最快多久）」，就必須用 BFS。**

這三類題型的選擇邏輯差異如下：

### 1. 為什麼「太平洋大西洋水流」與「島嶼個數」常用 DFS？

這兩題的本質都是 **Reachability（連通性 / 漫水填充 Flood Fill）**。我們只在乎「水能不能從邊界逆流到某個格子」或「哪些 `1` 連在一起」，完全不在乎走了幾步。

- **其實都能用 BFS 解**：太平洋大西洋水流問題（LC 417）如果把太平洋或大西洋的所有邊界點同時放進 Queue，用 **Multi-source BFS（多源 BFS）** 一樣可以解，時間複雜度與 DFS 完全相同，都是 $O(M \times N)$。
    
- **為何推薦 DFS**：在 2D Grid 純遍歷題型中，DFS 透過遞迴（Recursion）實作，不需要手動宣告 `deque` 和寫 `while queue:` 迴圈，程式碼行數少一半、面試時寫起來最快（唯一的限制是網格不能大到超出語言的 Recursion Stack 上限，例如 Python 預設 1000 層）。
    

### 2. 為什麼「騎士的最短路線」一定要用 BFS？

這題的本質是 **Unweighted Shortest Path（無權重圖的最短路徑）**，同類型的還有「腐爛的橘子（Rotting Oranges）」、「迷宮最短出口」。

- **BFS 的水波紋特性**：BFS 是「一層一層」向外擴散的。第 1 層是走 1 步能到的所有點，第 2 層是走 2 步能到的所有點。因此，**BFS 第一次碰到終點時，數學上保證就是最短步數**，可以直接提早結束（Early Return）。
    
- **DFS 的盲點**：DFS 會「一條路走到黑」。騎士可能會繞了很大一圈才碰到終點，而且第一次碰到時你並不知道這是不是最短的，必須把所有可能路徑都窮舉完並取 `min()`，時間複雜度會退化成指數級（Exponential），導致超時（TLE）。
    

### 3. 為什麼「拓撲排序（Topological Sort）」通常使用 BFS？

拓撲排序（例如課程表 Course Schedule）其實 **BFS（Kahn's Algorithm）和 DFS（後序遍歷反轉）都能解**，但實務與面試中幾乎一面倒推薦 BFS，原因有二：

1. **直覺貼合「解除依賴」的過程**：BFS 透過計算每個節點的 **入度（In-degree，代表還有幾門先修課沒修）**，先把 `in-degree == 0` 的節點放進 Queue。每修完一門課，就把後續課程的入度減 1，一旦變成 0 就解鎖放進 Queue。這種「剝洋蔥」的層次感天生就是 BFS。
    
2. **偵測環（Cycle Detection）極度簡單**：BFS 跑完後，只要檢查「處理過的節點總數是否等於總課程數」，就能立刻判斷圖中有沒有循環依賴；如果用 DFS，則必須額外維護三種節點狀態（`0: 未訪問`、`1: 正在當前路徑訪問中`、`2: 已完成訪問`）才能抓出環，實作細節更容易出錯。
    

### 題型與演算法選擇速查表

| **題目核心目標**        | **推薦演算法**              | **關鍵原因**                  | **經典代表題**                          |
| ----------------- | ---------------------- | ------------------------- | ---------------------------------- |
| **最少步數 / 最短時間**   | **BFS**                | 逐層擴散，首次抵達即為最優解            | 騎士最短路線、腐爛的橘子、單詞接龍                  |
| **區域數量 / 連通性**    | **DFS** (或 BFS)        | 不在乎距離，DFS 遞迴寫法最精簡         | 島嶼個數、太平洋大西洋水流、被圍繞的區域               |
| **窮舉所有路徑 / 排列組合** | **DFS (Backtracking)** | 需要一路走到底並透過狀態回溯來試錯         | 數獨、N 皇后、所有子集/排列                    |
| **依賴關係 / 排程順序**   | **BFS (Kahn's)**       | 透過 `In-degree` 逐層解鎖，判環最直觀 | 課程表 I & II、外星字典 (Alien Dictionary) |




#### lintcode的老師建議用BFS解題目
```
但為什麼lintcode的老師建議如果可以用BFS解的題目(可能也可以用DFS)儘量用BFS? 是否在coding interview用BFS比較恰當
```
LintCode（九章算法）最經典的口訣就是「能用 BFS 解決的，絕不用 DFS」**。這個建議在 Coding Interview 中非常務實，並不是因為 BFS 比較高級，而是基於**工程安全性**與**面試風險控管。

在面試中優先選用 BFS，主要有四個核心原因：

### 1. 徹底避開 Stack Overflow（遞迴爆棧）風險

這是面試官（尤其是 Senior 級別面試）最看重的工程細節：

- **DFS（遞迴版）用的是 Call Stack**：系統分配給函式呼叫的堆疊空間非常有限。以 Python 為例，預設的遞迴深度上限只有 **1,000 層**。如果面試官給一張 $100 \times 100$（10,000 個格子）的地圖，只要路徑呈蛇形繞行，遞迴 DFS 就會直接噴出 `RecursionError` 崩潰。
    
- **BFS 用的是 Heap（堆積記憶體）**：BFS 透過 `deque` 在記憶體的 Heap 區動態配置空間，只要系統 RAM 夠大就不會爆棧，在處理大規模資料（Large-scale input）時遠比遞迴安全。
    

### 2. 應對面試官 Follow-up（追問）的擴充性極高

面試題目經常是漸進式的：

- **第一問**：給你一張地圖，請問起點能不能走到終點？（連通性問題，DFS 和 BFS 都能解）
    
- **第二問（Follow-up）**：很好，那請問起點到終點**最少需要幾步**？或者如果有**多個起點同時擴散**呢？
    
- 如果你第一題寫的是 BFS，第二題只需要在 `while queue:` 裡加一個 `steps += 1` 就能直接交卷；如果你第一題寫的是 DFS，這時候就必須當場把整個架構打掉重寫，非常浪費寶貴的面試時間。
    

### 3. 寫法高度模板化，不容易在緊張時寫錯

- 遞迴 DFS 在處理「狀態恢復（Backtracking）」與「純遍歷（Visited）」時，初學者或緊張時很容易搞混 `visited` 的標記時機（例如不小心寫成離開節點時把 `visited` 設回 `False`，導致時間複雜度從 $O(V+E)$ 退化成指數級而 TLE）。
    
- BFS 的標準模板（`queue` + `visited` set，在**加入 Queue 的當下**就立刻標記 `visited`）結構非常固定，幾乎不會發生重複入隊或無窮迴圈的 Bug。
    

### 4. 既然要寫迴圈，不如直接寫 BFS

有人會說：「那我用 `Stack` 寫非遞迴（Iterative）的 DFS，不就也能避開 Stack Overflow 了嗎？」 這完全正確，但從實作成本來看，既然你都已經要寫 `while` 迴圈並宣告一個容器了，把 `stack.pop()` 換成 `queue.popleft()` 程式碼字數完全一樣，卻能額外免費獲得「最短路徑」的保證，性價比最高。

### 面試時的實戰判斷準則

雖然「能用 BFS 就用 BFS」是很好的防禦性策略，但在面試中可以依題型做微調，展現你對不同資料結構的掌握度：

- **Graph / 2D Matrix（圖論與網格題）**：**無腦首選 BFS**。不管是算島嶼數量、最短路徑還是拓撲排序，用 BFS 最穩健，也能展現你對記憶體與規模化（Scalability）的考量。
    
- **Binary Tree（二元樹題）**：**優先用遞迴 DFS**。一般樹的高度 $H$ 通常是 $O(\log N)$（除非退化成鏈表），爆棧機率低，且樹的 Divide & Conquer（分治法）用遞迴寫起來只有 5~10 行，可讀性遠勝 BFS。
    
- **組合 / 排列 / 搜尋所有解（Backtracking）**：**必須用 DFS**。這類題目需要維護路徑狀態並「回溯試錯」，BFS 無法有效處理這類狀態分支。




#### 遞迴爆棧跟DFS用Stack, BFS用deque跟Heap關係
```
我不太理解你說的徹底避開 Stack Overflow（遞迴爆棧）風險. 因為DFS用Stack資料結構是後進先出, BFS用deque是先進先出, 那Heap應該是一種樹狀結構通常使用在最大堆最小堆維護最大最小, 為什麼說Heap是堆積記憶體? 那跟BFS的關係是甚麼? 為什麼DFS就會有Stack Overflow風險這是甚麼? 請詳細說明並用圖輔助說明並具體舉例說明
```
這是一個非常經典的觀念混淆點，原因在於「資料結構（Data Structure）」**與**「作業系統記憶體配置（Memory Layout）」剛好使用了兩個相同的英文單字：**Stack** 與 **Heap**。

簡單來說：**資料結構的 Heap（最大堆／最小堆）跟記憶體的 Heap（堆積記憶體），除了英文名字撞名之外，兩者完全沒有關係。**

如上圖所示，當程式執行時，作業系統會把記憶體切成兩大核心工作區（左側 **Stack** 與右側 **Heap**）。以下從「名詞對照」、「BFS 與 Heap 的關係」以及「DFS 爆棧的具體過程」三個層面拆解說明：

### 1. 名詞撞名對照：資料結構 vs. 記憶體分區

|**英文名詞**|**在「演算法／資料結構」中的意思**|**在「作業系統／記憶體配置」中的意思**|
|---|---|---|
|**Stack**|**堆疊資料結構**：後進先出（LIFO）的容器，例如 Python 的 `list` 搭配 `append()` / `pop()`。|**函式呼叫棧（Call Stack）**：一塊**空間極小、大小固定**的連續記憶體（通常僅 1 MB ～ 8 MB），專門用來記錄「目前函式執行到哪一層、區域變數與返回位址」。|
|**Heap**|**堆積（優先佇列）**：一種完全二元樹，用來維護最大值或最小值（如 Python 的 `heapq`）。|**堆積記憶體（Heap Memory）**：一塊**空間極大、可動態擴充**的自由記憶體池（大小受限於系統 RAM，可達數 GB）。**所有程式動態建立的物件本體都存放在這裡。**|
![[Pasted image 20261127163607.png|393]]

### 2. 為什麼說 BFS 用的 `deque` 在「Heap 記憶體」裡？

當你在寫 BFS 並宣告 `queue = deque()` 時，底層記憶體發生了兩件事（對應上方圖片的左右兩側）：

1. **左側 Call Stack**：只在當前函式的 Stack Frame 裡，放了一個極小的指標變數（Reference）叫做 `queue`。
    
2. **右側 Heap Memory**：`deque` 這個容器物件本身，以及你不斷 `append((nr, nc))` 塞進去的成千上萬個座標，**全部都是動態配置在空間極大的 Heap 記憶體裡**。

因為 BFS 是透過 `while queue:` 迴圈在同一個函式內運作，函式從頭到尾都沒有呼叫自己，所以左側 **Call Stack 的深度永遠維持在 1 層**；即使地圖有 100 萬個格子，只要你的電腦 RAM（Heap）裝得下，程式就絕對不會崩潰。

### 3. 為什麼遞迴 DFS 會「爆棧（Stack Overflow）」？具體舉例

遞迴（Recursion）的本質是：**外層函式還沒執行完，就暫停在原地並呼叫內層函式**。

因為外層函式還沒結束，CPU **不能釋放**它的記憶體，必須把當下的區域變數（例如目前的座標 `r, c`、迴圈走到哪個方向 `dr, dc`）打包成一個 **Stack Frame（函式框架）**，像疊盤子一樣壓進左側狹小的 **Call Stack** 中，等內層函式全部結束後才能回來繼續執行。

#### 具體數字推演：一張 $50 \times 50$ 的「蛇形迷宮」

假設面試官給了一張 $50 \times 50$（共 **2,500 個格子**）的地圖（例如 LeetCode 200 島嶼個數，整張地圖都是相連的陸地 `'1'`），我們用遞迴 DFS 遍歷：
```python
def dfs(r, c):
    visited.add((r, c))
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if valid(nr, nc) and (nr, nc) not in visited:
            dfs(nr, nc)  # <-- 卡在這裡！必須等下一層全部跑完，這一行才算結束
```

這 2,500 個相連的格子會導致 Call Stack 發生以下連鎖反應：

1. **第 1 層**：呼叫 `dfs(0, 0)`，執行到第 6 行呼叫 `dfs(0, 1)` $\rightarrow$ `dfs(0, 0)` **暫停並留在 Call Stack**。
    
2. **第 2 層**：呼叫 `dfs(0, 1)`，執行到第 6 行呼叫 `dfs(0, 2)` $\rightarrow$ `dfs(0, 1)` **暫停並留在 Call Stack**。
    
3. **第 3 層 ～ 第 999 層**：一路沿著蛇形路徑往深處鑽，Call Stack 裡已經疊了 999 個尚未結束的函式框架。
    
4. **第 1,000 層（崩潰點）**：當走到第 1,000 個相連格子 `dfs(19, 49)` 時，觸發 Python 保護 Call Stack 的硬性上限（預設 `sys.getrecursionlimit() == 1000`），系統直接拋出 **`RecursionError: maximum recursion depth exceeded`** 強制終止程式（在 C/C++ 中則會直接踩破記憶體邊界，引發 **Stack Overflow / Segmentation Fault**）。
    



#### 常用演算法與資料結構的記憶體分布
```
所以其他的leetcode/lintcode常用算法除了DFS有用遞迴之外譬如two pointers, binary search或binary search tree都不會有stack overflow因為主要data都放在記憶體的heap, 而function指針放在stack通常都是有限的? 而當我們在解法使用到list, deque, stack, hash map, binary search tree, graph等資料結構都是放在記憶體的Heap裡?

也示範如何用顯示Stack+heap記憶體寫出不會爆棧的iterative DFS, 所以顯式跟隱式Stack差別在哪裡? 所以sys.setrecursionlimit()具體作用是增加記憶體的Stack嗎? 以及詳細解釋對比C++, python在call stack與heap在記憶體配置上的差異
```
你的理解非常精準，核心邏輯完全正確，只有針對 **Binary Search Tree（二元搜尋樹）** 需要補充一個關鍵例外：**樹的「節點資料」確實全在 Heap，但如果你用「遞迴」去遍歷一棵極度不平衡的樹，一樣會發生 Stack Overflow。**

以下逐一針對你的四個問題拆解說明：

### 1. 其他常用演算法與資料結構的記憶體分布

- **Two Pointers（雙指針）與迭代版 Binary Search（二分搜尋）**： 這兩者只用 `while` 迴圈控制索引變數（如 `left`、`right`），函式從頭到尾只佔用 **1 層 Call Stack**，因此**永遠不會** Stack Overflow。
    
- **`list`、`deque`、`stack`、`dict`（Hash Map）、`set`、Graph / Tree 節點**： 在 Python 中，**所有容器物件與節點實例本體 100% 都配置在 Heap Memory**；Call Stack 上只存放指向這些 Heap 物件的「參考指標（Reference）」。
    
- **Binary Search Tree（BST）的唯一陷阱**：
    
    - **平衡樹（Balanced BST）**：樹高 $H = O(\log N)$。就算有 100 萬個節點，樹高也只有約 20 層，遞迴遍歷只佔用 20 層 Call Stack，非常安全。
        
    - **退化成鏈表的偏斜樹（Skewed BST）**：如果依序把 `1, 2, 3, ..., 10000` 插入未平衡的 BST，樹會變成一條直線，樹高 $H = O(N)$。此時若用遞迴做搜尋或遍歷，Call Stack 會疊到 10,000 層而觸發 `RecursionError`（解決方法是改用 `while curr:` 迭代版）。
        

### 2. 顯式 Stack vs. 隱式 Stack 與 Iterative DFS 範例

要寫出「永遠不會爆棧」的 DFS，核心技巧就是：**不要讓函式呼叫自己，而是自己宣告一個存放於 Heap 的 `list` 作為 Stack，搭配 `while` 迴圈模擬遞迴。**
```Python
from typing import List

def iterative_dfs(grid: List[List[str]], start_r: int, start_c: int) -> int:
    rows, cols = len(grid), len(grid[0])
    # 1. 顯式 Stack：這個 list 物件配置在空間極大的 Heap Memory 中
    stack = [(start_r, start_c)]
    visited = set()
    area = 0
    
    # 反向排列方向，可讓彈出順序與遞迴版 DFS 完全一致
    DIRECTIONS = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    
    # 2. 用 while 迴圈取代函式自我呼叫，Call Stack 深度永遠維持在 1 層
    while stack:
        r, c = stack.pop()  # LIFO：從尾端彈出最新加入的節點
        
        if (r, c) in visited:
            continue
        visited.add((r, c))
        area += 1
        
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows 
                and 0 <= nc < cols 
                and grid[nr][nc] == "1" 
                and (nr, nc) not in visited
            ):
                stack.append((nr, nc))  # 壓入顯式 Stack
                
    return area
```

#### 隱式堆疊（Implicit Stack）vs. 顯式堆疊（Explicit Stack）

|**比較維度**|**隱式堆疊（Implicit Stack / 遞迴 DFS）**|**顯式堆疊（Explicit Stack / 迭代 DFS）**|
|---|---|---|
|**實體位置**|作業系統配置的 **Call Stack**（空間極小，1 MB ~ 8 MB）|動態配置的 **Heap Memory**（受限於系統 RAM，可達數 GB）|
|**維護者**|由作業系統與語言直譯器在底層自動建立與銷毀 Stack Frame|由工程師自己在程式碼中宣告 `stack = []` 並手動 `append` / `pop`|
|**儲存內容**|整個函式的完整上下文（返回位址、所有區域變數、迴圈狀態）|僅儲存你需要的最小狀態（例如只有 `(r, c)` 座標 Tuple，極省記憶體）|
|**爆棧風險**|**有**（深度超過上限直接拋出 `RecursionError` 或 Crash）|**無**（只要系統記憶體夠用就不會爆）|

### 3. `sys.setrecursionlimit()` 是在增加記憶體的 Stack 嗎？

**不是！它只是調高 Python 直譯器的「軟體煞車線」，完全沒有向作業系統申請更大的硬體 Call Stack 記憶體。**

Python 的遞迴保護其實有兩道關卡：

1. **第一道（Python 軟體保護閥）**：CPython 為了避免無限遞迴直接把整個程式弄崩潰，內建了一個計數器（預設上限 `1000`）。當遞迴深度達到 1,000 層時，Python 會主動踩煞車並拋出可捕捉的 `RecursionError`。呼叫 `sys.setrecursionlimit(10000)` 只是把這個計數器的警報線往後移。
    
2. **第二道（作業系統 OS Thread Stack 硬上限）**：作業系統分配給每個執行緒（Thread）的實體 Call Stack 記憶體大小是固定的（Windows 預設通常為 **1 MB**，Linux / macOS 通常為 **8 MB**）。

如果你把 `setrecursionlimit` 設得過大（例如 `100000`），在還沒撞到 Python 軟體上限之前，實體的 1 MB / 8 MB Call Stack 記憶體就會先被耗盡。這時作業系統會直接觸發 **Segmentation Fault（記憶體區段錯誤）** 強制殺掉整個 Python 程序，連 `try...except` 都救不回來。因此面試中**不建議**依賴 `setrecursionlimit()`。

### 4. C++ 與 Python 在 Call Stack 與 Heap 記憶體配置的差異

這兩種語言在記憶體模型上有本質上的設計差異：**Python 採取「萬物皆物件（All on Heap）」策略，而 C++ 採取「值語義（Value Semantics，預設放 Stack、按需放 Heap）」策略。**

|**比較項目**|**Python (CPython)**|**C++**|
|---|---|---|
|**基本型別變數**<br><br>  <br><br>（如 `int x = 10`）|`10` 是一個完整的 `PyLongObject` 放在 **Heap**；**Call Stack** 裡只放一個指向它的指標（Reference）。|`10`（4 bytes）**直接塞在 Call Stack** 的 Stack Frame 記憶體裡，沒有額外指標開銷。|
|**固定大小陣列**|Python 沒有原生 Stack 陣列，`list` 本體與元素全在 **Heap**。|`int arr[1000];` **直接配置在 Call Stack** 上（若開 `int arr[10000000];` 40 MB 會立刻 Stack Overflow）。|
|**標準容器**<br><br>  <br><br>（`list`/`deque` vs. `vector`/`queue`/`stack`）|容器結構與元素全部都在 **Heap**。|**混合配置**：容器的控制標頭（如 `data` 指標、`size`、`capacity`，約 24 bytes）在 **Call Stack**；但內部儲存的**大量元素陣列全部動態配置在 Heap**。|
|**自定義物件 / 樹節點**|`node = TreeNode(5)` 永遠配置在 **Heap**。|可自由選擇：`TreeNode node(5);` 放在 **Call Stack**；`new TreeNode(5)` 或 `std::make_unique<TreeNode>(5)` 放在 **Heap**。|
|**記憶體釋放機制**|由引用計數（Reference Counting）+ 垃圾回收器（GC）自動管理 Heap 釋放。|**Stack 變數**在離開 `{}` 作用域時自動瞬間銷毀；**Heap 記憶體**需透過智慧指標（RAII）或手動 `delete` 釋放。|

#### C++ 容器的常見誤解補充

很多人在寫 C++ 時會擔心：「我在函式裡宣告 `std::queue<int> q;` 或 `std::stack<int> st;`，並往裡面塞了 100 萬個節點，這樣會把 Call Stack 塞爆嗎？」

**答案是不會。** 因為 C++ STL 容器（`std::vector`、`std::deque`、`std::queue`、`std::stack`、`std::unordered_map`）在 Call Stack 上只佔用幾十個 Bytes 的管理指標，背後真正用來裝 100 萬個資料的記憶體區塊，全都是 STL 透過配置器（`std::allocator`）向 **Heap Memory** 申請的。