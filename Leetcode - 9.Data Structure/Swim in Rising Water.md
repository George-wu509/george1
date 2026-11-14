
水位上升的泳池中游泳 Swim in Rising Water

Leetcode 778 / Lintcode 1031 在一个 `n x n` 的整数矩阵 `grid` 中，每一个方格的值 `grid[i][j]` 表示位置 `(i, j)` 的平台高度。

当开始下雨时，在时间为 `t` 时，水池中的水位为 `t` 。你可以从一个平台游向四周相邻的任意一个平台，但是前提是此时水位必须同时淹没这两个平台。假定你可以瞬间移动无限距离，也就是默认在方格内部游动是不耗时的。当然，在你从平台出发时，你必须站在边界之内。

请你从坐标方格的左上平台 `(0, 0)` 出发，最少耗时多少时间你才能到达坐标方格的右下平台 `(n - 1, n - 1)`？

**样例 1：** 输入：
```Python
grid = [[0,2],[1,3]]
```

输出：
```Python
3
```

解释： 时间为 0 时，你位于坐标方格的位置为 `(0, 0)`。 此时你不能游向任意方向，因为四个相邻方向平台的高度都大于当前时间为 0 时的水位。 等时间到达 3 时，你才可以游向平台 `(1, 1)`。因为此时的水位是 3，坐标方格中的平台没有比水位 3 更高的，所以你可以游向坐标方格中的任意位置。

**样例 2：** 输入：
```Python
grid = [[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]
```

输出：
```Python
16
```

解释： 最终的路线为：`0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 16 -> 15 -> 14 -> 13 -> 12 -> 11 -> 10 -> 9 -> 8 -> 7 -> 6`。 必须等到时间为 16 时，水位才能淹没 `(2, 4)` 处的平台 `16`，从而连通整条路径到达右下角 `(4, 4)`。

**样例 3（起点高度大于 0 的情况）：** 输入：
```Python
grid = [[3,2],[0,1]]
```

输出：
```Python
3
```

解释： 因为起点 `(0, 0)` 的高度本身就是 `3`，必须等到时间 `t = 3` 水位淹没起点后才能出发，之后水位 3 足以直接游到 `(1, 1)`。

```Python
import heapq
from typing import List

class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        n = len(grid)
        # 最小堆存储 (当前路径上的最大高度, r, c)
        min_heap = [(grid[0][0], 0, 0)]
        visited = {(0, 0)}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while min_heap:
            t, r, c = heapq.heappop(min_heap)
            if r == n - 1 and c == n - 1:
                return t
                
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    heapq.heappush(min_heap, (max(t, grid[nr][nc]), nr, nc))
                    
        return -1
```


```Python
from typing import List

class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        n = len(grid)
        parent = list(range(n * n))
        
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
            
        def union(x: int, y: int):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y
                
        # 记录每个高度对应的坐标位置 (heights[val] = (r, c))
        pos = [None] * (n * n)
        for r in range(n):
            for c in range(n):
                pos[grid[r][c]] = (r, c)
                
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # 按时间 t 从 0 到 n^2 - 1 逐步升高水位
        for t in range(n * n):
            r, c = pos[t]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                # 若相邻格子高度不超过当前水位 t，说明已被淹没，可以连通
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] <= t:
                    union(r * n + c, nr * n + nc)
            # 检查起点 (0, 0) 与终点 (n-1, n-1) 是否已连通
            if find(0) == find(n * n - 1):
                return t
                
        return -1
```

解釋: step1: 題目求從 `(0, 0)` 到 `(n-1, n-1)` 的路徑中，**路徑上最大平台高度的最小值**（瓶頸最短路）。 step2: 以 `grid = [[0, 2], [1, 3]]` 為例，初始化最小堆 `min_heap = [(0, 0, 0)]`，`visited = {(0, 0)}`。 step3: 彈出 `(0, 0, 0)`，檢查四個方向：右邊 `(0, 1)` 高度為 `2`，入堆 `(max(0, 2), 0, 1) = (2, 0, 1)`；下邊 `(1, 0)` 高度為 `1`，入堆 `(max(0, 1), 1, 0) = (1, 1, 0)`。 step4: 堆中現有 `[(1, 1, 0), (2, 0, 1)]`，貪心彈出水位要求最小的 `(1, 1, 0)`。 step5: 從 `(1, 0)` 檢查未訪問鄰居，發現右下角 `(1, 1)` 高度為 `3`，計算到達該點的瓶頸水位 `max(1, 3) = 3`，入堆 `(3, 1, 1)`。 step6: 接著彈出 `(2, 0, 1)`，其鄰居 `(1, 1)` 已在 `visited` 中（因為之前從更優或同級路徑入過隊），跳過。 step7: 最後彈出 `(3, 1, 1)`，發現已抵達終點 `(1, 1)`，直接返回當前路徑最大水位 `3`。

解釋: 因為在方格間移動不耗時，只要水位 `t` 大於等於路徑上**所有格子的高度**，就能瞬間從起點游到終點。 因此問題等價於：**尋找一條從 `(0, 0)` 到 `(n-1, n-1)` 的路徑，使得路徑上的最大高度 `max(grid[r][c])` 最小**。 狀態轉移公式（Modified Dijkstra）：`dist[nr][nc] = min(dist[nr][nc], max(dist[r][c], grid[nr][nc]))`。

### **LeetCode 778 - Swim in Rising Water（水位上升的泳池中游泳）**

這道題目是圖論中非常經典的 **瓶頸最短路徑（Minimax Path）** 問題。與傳統最短路徑「求邊權之和最小」不同，本題要求「路徑上的最大節點權值最小」，常見解法有 **Dijkstra 變體（最小堆貪心）**、**並查集（Union-Find）** 以及 **二分答案 + BFS/DFS**。

## **解法解析**

### **思路**

1. **問題轉化（抽離時間概念）**
    
    - 題目說「每過 1 單位時間水位上升 1，且移動瞬間完成」，這意味著我們根本不需要一步步模擬時間流逝。
        
    - 只要選定一條路徑，這條路徑需要的最少等待時間，就是**這條路徑上最高的那個平台高度**（同時不能小於起點高度 `grid[0][0]`）。
        
    - 問題轉化為：在網格圖中找一條從左上角到右下角的路徑，使**路徑經過的最大值最小**。
        
2. **Dijkstra 貪心擴展（最小堆）**
    
    - 維護一個最小堆 `min_heap`，存放三元組 `(max_height_so_far, r, c)`，表示從起點走到 `(r, c)` 所需經歷的最小水位。
        
    - 每次從堆中彈出當前所需水位**最低**的格子進行擴展：因為每次都是挑「門檻最低」的格子往外走，所以當第一次彈出終點 `(n-1, n-1)` 時，所攜帶的水位就一定是全局最小答案。
        
    - 由於每個格子第一次進入堆時，就已經是由當前最小水位邊界觸及的（所有格子高度互不相同且邊權由節點高度決定），因此一旦標記 `visited` 就不需要二次更新，每個格子最多入堆出堆一次。
        
3. **代碼執行流程**
    
    - 將起點 `(grid[0][0], 0, 0)` 放入 `min_heap`，並加入 `visited`。
        
    - 迴圈彈出堆頂元素 `(t, r, c)`：
        
        - 若 `(r, c) == (n-1, n-1)`，直接返回 `t`。
            
        - 遍歷上下左右四個相鄰格子 `(nr, nc)`，若未訪問過，標記為已訪問並將 `(max(t, grid[nr][nc]), nr, nc)` 推入堆中。
            

## **具體舉例**

假設 `grid = [[0, 2], [1, 3]]`，Dijkstra 最小堆執行過程如下：

|**步驟**|**彈出節點 (t, r, c)**|**探索未訪問鄰居**|**min_heap 狀態**|**visited 集合**|
|---|---|---|---|---|
|初始化|-|起點 `(0, 0)` 入堆|`[(0, 0, 0)]`|`{(0,0)}`|
|Step 1|`(0, 0, 0)`|`(0,1)` 高度 2 → `max(0,2)=2`<br><br>  <br><br>`(1,0)` 高度 1 → `max(0,1)=1`|`[(1, 1, 0), (2, 0, 1)]`|`{(0,0), (0,1), (1,0)}`|
|Step 2|`(1, 1, 0)`|`(1,1)` 高度 3 → `max(1,3)=3`|`[(2, 0, 1), (3, 1, 1)]`|`{(0,0), (0,1), (1,0), (1,1)}`|
|Step 3|`(2, 0, 1)`|無未訪問鄰居|`[(3, 1, 1)]`|`{(0,0), (0,1), (1,0), (1,1)}`|
|Step 4|`(3, 1, 1)`|**抵達終點 `(1, 1)`**，返回 `3`|`[]`|`{(0,0), (0,1), (1,0), (1,1)}`|

最終返回 `3`。

## **時間與空間複雜度分析**

設網格邊長為 $N$，則總格子數 $V = N^2$。

- **時間複雜度：$O(N^2 \log N)$**
    
    - 網格中共有 $N^2$ 個格子，每個格子最多入堆和出堆一次。
        
    - 堆的大小最多為 $O(N^2)$，每次 `heappush` 和 `heappop` 操作的時間複雜度為 $O(\log(N^2)) = O(2 \log N) = O(\log N)$。
        
    - 總時間複雜度為 **$O(N^2 \log N)$**。
        
- **空間複雜度：$O(N^2)$**
    
    - `visited` 集合與最小堆 `min_heap` 最多需要存儲全部 $N^2$ 個格子的狀態，因此空間複雜度為 **$O(N^2)$**。
        

## **其他解法**

### **1. 並查集（Union-Find / Kruskal 最小生成樹思想）**

- **思路**：
    
    - 題目保證 `grid[i][j]` 是 `0` 到 $N^2 - 1$ 的一個排列。
        
    - 我們可以模擬時間 `t` 從 `0` 到 $N^2 - 1$ 逐秒上升：在時間 `t`，找到高度剛好為 `t` 的平台，將它與上下左右**高度小於等於 `t`**（已經被水淹沒）的相鄰平台用並查集合併。
        
    - 每合併完一次，檢查起點 `0` 與終點 `N^2 - 1` 是否屬於同一個連通分量（`find(0) == find(N*N - 1)`），第一次連通時的 `t` 即為答案。
        
- **時間複雜度**：$O(N^2 \alpha(N^2)) \approx O(N^2)$（因為高度範圍固定為 $0 \sim N^2-1$，可用桶排序思想 $O(1)$ 定位格子，無需 $O(N^2 \log N)$ 排序）。
    
- **空間複雜度**：$O(N^2)$。
    

### **2. 二分答案 + BFS / DFS**

- **思路**：
    
    - 答案具有單調性：如果水位 `mid` 能游到終點，那麼任何大於 `mid` 的水位也一定能游到終點。
        
    - 二分範圍設為 `left = max(grid[0][0], grid[n-1][n-1])`，`right = n * n - 1`。
        
    - 每次猜測水位 `mid`，只允許走高度 `<= mid` 的格子，用 BFS 或 DFS 判斷能否從 `(0, 0)` 走到 `(n-1, n-1)`，收縮邊界直到找到最小合法水位。
        
- **時間複雜度**：$O(N^2 \log N)$（二分 $\log(N^2)$ 次，每次 BFS/DFS 耗時 $O(N^2)$）。
    
- **空間複雜度**：$O(N^2)$。
    

## **總結**

- **最推薦面試寫法**：**Dijkstra 變體（最小堆）**，代碼簡短、邏輯直觀，且不依賴「高度必須是 $0 \sim N^2 - 1$ 排列」的特殊條件，通用性最強。
    
- **理論時間最優解**：**並查集（Union-Find）**，利用數值排列特性省去排序，時間複雜度可達接近線性的 $O(N^2 \alpha(N^2))$。
    
- **思維最易理解解法**：**二分答案 + BFS/DFS**，將「求最優值」轉化為「判定性問題」，適合在想不到圖論最優演算法時快速寫出。
    

## **圖論 / 瓶頸路徑與網格連通性類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 778**|Swim in Rising Water（水位上升的泳池中游泳）|**網格瓶頸最短路（路徑最大值最小）**|**Dijkstra（最小堆） / 並查集 / 二分 + BFS**|
|**LeetCode 1631**|Path With Minimum Effort（最小體力消耗路徑）|**網格瓶頸最短路（相鄰高度差最大值最小）**|**Dijkstra（最小堆） / 並查集（Kruskal） / 二分 + BFS**|
|**LeetCode 1102**|Path With Maximum Minimum Value（得分最高的路徑）|**網格瓶頸最長路（路徑最小值最大）**|**Dijkstra（最大堆） / 並查集（從大到小合併）**|
|**LeetCode 743**|Network Delay Time（網絡延遲時間）|**有向加權圖單源最短路徑（邊權和最小）**|**標準 Dijkstra 算法（最小堆）**|
|**LeetCode 1514**|Path with Maximum Probability（概率最大的路徑）|**無向圖最大乘積路徑**|**Dijkstra 變體（最大堆，邊權相乘）**|
|**LeetCode 1584**|Min Cost to Connect All Points（連接所有點的最小費用）|**無向完全圖最小生成樹（MST）**|**Prim 算法 / Kruskal 算法 + 並查集**|
|**LeetCode 1135**|Connecting Cities With Minimum Cost（最低成本聯通所有城市）|**無向圖最小生成樹（MST）**|**Kruskal 算法 + 並查集（按邊權排序）**|
|**LeetCode 407**|Trapping Rain Water II（接雨水 II）|**2D 網格木桶效應 / 邊界向內收縮**|**最小堆（從外圍一圈向內部 BFS 維護水位）**|

**進階變形題（LeetCode 1631 - Path With Minimum Effort）：** 你準備參加一場遠足活動。給你一個二維 `rows x columns` 的地圖 `heights` ，其中 `heights[row][col]` 表示格子 `(row, col)` 的高度。一開始你在最左上角的格子 `(0, 0)` ，且你希望去最右下角的格子 `(rows-1, columns-1)`。你每次可以往 上，下，左，右 四個方向之一移動，你想要找到耗費 **體力** 最小的一條路徑。一條路徑耗費的 **體力值** 是路徑上相鄰格子之間 **高度差絕對值** 的 **最大值** 決定的。請你返回從左上角走到右下角的最小 **體力消耗值** 。

Python

```
import heapq
from typing import List

class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        rows, cols = len(heights), len(heights[0])
        # efforts[r][c] 记录到达 (r, c) 所需的最小体力消耗
        efforts = [[float('inf')] * cols for _ in range(rows)]
        efforts[0][0] = 0
        
        min_heap = [(0, 0, 0)]  # (当前路径最大高度差, r, c)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while min_heap:
            curr_effort, r, c = heapq.heappop(min_heap)
            if r == rows - 1 and c == cols - 1:
                return curr_effort
            # 若已有更优路径到达该点，跳过过期状态
            if curr_effort > efforts[r][c]:
                continue
                
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    next_effort = max(curr_effort, abs(heights[nr][nc] - heights[r][c]))
                    if next_effort < efforts[nr][nc]:
                        efforts[nr][nc] = next_effort
                        heapq.heappush(min_heap, (next_effort, nr, nc))
                        
        return 0
```