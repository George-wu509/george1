

K 站中轉內最便宜的航班 Cheapest Flights Within K Stops

Leetcode 787 / Lintcode 1029 有 `n` 个城市通过一些航班连接。给你一个数组 `flights` ，其中 `flights[i] = [from_i, to_i, price_i]` ，表示该航班都从城市 `from_i` 开始，以价格 `price_i` 抵达 `to_i`。

现在给定所有的城市和航班，以及出发城市 `src` 和目的地 `dst`，你的任务是找到出一条最多经过 `k` 站中转的路线，使得从 `src` 到 `dst` 的 **价格最便宜** ，并返回该价格。如果不存在这样的路线，则输出 `-1`。

**样例 1：** 输入：
```Python
n = 4
flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]
src = 0
dst = 3
k = 1
```

输出：

```Python
700
```

解释：城市航线图中，从城市 `0` 到城市 `3` 最多经过 `1` 站中转的最便宜路线是 `0 -> 1 -> 3` ，费用为 `100 + 600 = 700`。 注意：`0 -> 1 -> 2 -> 3` 虽然总费用更便宜（`100 + 100 + 200 = 400`），但经过了 `1` 和 `2` 共 `2` 站中转，超过了 `k = 1` 的限制。

**样例 2：** 输入：
```Python
n = 3
flights = [[0,1,100],[1,2,100],[0,2,500]]
src = 0
dst = 2
k = 1
```

输出：
```Python
200
```

解释： 从城市 `0` 到城市 `2` 最多经过 `1` 站中转的最便宜路线是 `0 -> 1 -> 2` ，费用为 `100 + 100 = 200`（优于直飞的 `500`）。

**样例 3（无法在 K 站内抵达的情况）：** 输入：

```Python
n = 3
flights = [[0,1,100],[1,2,100],[0,2,500]]
src = 0
dst = 2
k = 0
```

输出：

```Python
500
```

解释： `k = 0` 表示不允许任何中转（只能直飞，最多搭乘 `1` 趟航班），因此只能选择 `0 -> 2` ，费用为 `500`。

```Python
from typing import List

class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # Bellman-Ford 算法：最多经过 k 站中转 = 最多搭乘 k + 1 条边（航班）
        prices = [float("inf")] * n
        prices[src] = 0
        
        for _ in range(k + 1):
            # 必须基于上一轮的 prices 进行松弛，防止同一轮内串联更新多条边
            temp_prices = prices.copy()
            for u, v, cost in flights:
                if prices[u] != float("inf") and prices[u] + cost < temp_prices[v]:
                    temp_prices[v] = prices[u] + cost
            prices = temp_prices
            
        return prices[dst] if prices[dst] != float("inf") else -1
```


```Python
from collections import defaultdict, deque
from typing import List

class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        graph = defaultdict(list)
        for u, v, cost in flights:
            graph[u].append((v, cost))
            
        # min_cost[i] 记录目前到达城市 i 的最低花费
        min_cost = [float("inf")] * n
        min_cost[src] = 0
        
        # 队列存储 (当前城市, 累计花费)，按层（航班数）扩散
        queue = deque([(src, 0)])
        stops = 0
        
        while queue and stops <= k:
            for _ in range(len(queue)):
                curr_city, curr_cost = queue.popleft()
                for next_city, price in graph[curr_city]:
                    # 只有当新路径更便宜时才入队剪枝
                    if curr_cost + price < min_cost[next_city]:
                        min_cost[next_city] = curr_cost + price
                        queue.append((next_city, min_cost[next_city]))
            stops += 1
            
        return min_cost[dst] if min_cost[dst] != float("inf") else -1
```

解釋: step1: 「最多經過 `k` 站中轉」等價於從起點 `src` 到終點 `dst` 最多搭乘 `k + 1` 趟航班（即路徑最多包含 `k + 1` 條邊）。 step2: 以 `n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1` 為例，初始化 `prices = [0, inf, inf, inf]`，需要執行 `k + 1 = 2` 輪鬆弛。 step3: 第 1 輪（搭乘 1 趟航班）：複製 `temp_prices = [0, inf, inf, inf]`。遍歷所有航班，只有從 `0` 出發的 `[0, 1, 100]` 能更新 `temp_prices[1] = 100` -> `prices = [0, 100, inf, inf]`。 step4: 第 2 輪（搭乘 2 趟航班，即中轉 1 次）：複製 `temp_prices = [0, 100, inf, inf]`。遍歷所有航班：

- `[1, 2, 100]` 更新 `temp_prices[2] = 100 + 100 = 200`
    
- `[1, 3, 600]` 更新 `temp_prices[3] = 100 + 600 = 700`
    
- `[2, 3, 200]` 因為讀取的是上一輪 `prices[2] = inf`，所以不會在同輪誤用刚更新的 `200`！ step5: 兩輪結束後 `prices = [0, 100, 200, 700]`，返回 `prices[3] = 700`。
    

解釋: 這題本質上是**帶邊數限制的單源最短路徑問題**。 若直接套用標準 Dijkstra 算法，可能會為了追求「當前總花費最低」而走了太多中轉站，導致後續超出 `k` 站限制而被迫丟棄；而 **Bellman-Ford 算法** 每一輪迭代恰好對應「多走一條邊」，天生契合「最多 `k + 1` 條邊」的限制。 動態規劃狀態轉移方程（最多 `i` 條邊到達 `v` 的最低費用）： `dp[i][v] = min(dp[i-1][v], min(dp[i-1][u] + cost(u, v)))`。

### **LeetCode 787 - Cheapest Flights Within K Stops（K 站中轉內最便宜的航班）**

這道題目是圖論中非常經典的 **帶邊數限制的最短路徑（Constrained Shortest Path）** 問題。題目給定有向加權圖與中轉站數量上限 `k`，要求從 `src` 到 `dst` 的最小總花費，標準解法為 **Bellman-Ford 算法（滾動陣列優化）** 或 **按層擴展的 BFS**。

## **解法解析**

### **思路**

1. **問題轉化（中轉站數 → 邊數限制）**
    
    - 題目限制「最多經過 `k` 站中轉」，如果起點與終點之間經過了 `k` 個中間城市，那麼總共搭乘的航班數（圖中的有向邊數）就是 **`k + 1` 條**。
        
    - 因此，原問題完全等價於：**在有向加權圖中，尋找一條從 `src` 到 `dst` 且邊數不超過 `k + 1` 的最短路徑**。
        
2. **為何需要「複製上一輪陣列 `temp_prices`」**
    
    - 在標準的 Bellman-Ford 算法中，通常會直接在一維陣列 `prices` 上原地更新並重複 `n - 1` 次。
        
    - 但在本題中，如果第 1 輪剛更新了 `prices[1] = 100`，同一輪後面的邊 `[1, 2, 100]` 若直接讀取新的 `prices[1]`，就會在「1 輪之內」連續走完兩條邊（發生串聯更新），破壞「第 `i` 輪最多走 `i` 條邊」的性質。
        
    - 因此每輪迭代前必須先備份 `temp_prices = prices.copy()`，確保所有更新都基於**上一輪的狀態**。
        
3. **代碼執行流程**
    
    - 初始化長度為 `n` 的 `prices` 陣列，全部設為無窮大 `inf`，僅將起點 `prices[src]` 設為 `0`。
        
    - 執行 `k + 1` 次外層迴圈：
        
        - 複製一份 `temp_prices = prices.copy()`。
            
        - 遍歷每一條航班 `[u, v, cost]`，若 `prices[u]` 可達且 `prices[u] + cost < temp_prices[v]`，則更新 `temp_prices[v]`。
            
        - 本輪結束後令 `prices = temp_prices`。
            
    - 迴圈結束後，若 `prices[dst]` 仍為 `inf` 則返回 `-1`，否則返回 `prices[dst]`。
        

## **具體舉例**

假設 `n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1`，Bellman-Ford 執行 `k + 1 = 2` 輪的陣列變化如下：

|**迭代輪次**|**允許最多邊數**|**prices[0]**|**prices[1]**|**prices[2]**|**prices[3]**|**本輪有效更新說明**|
|---|---|---|---|---|---|---|
|初始化|`0` 條邊|`0`|`inf`|`inf`|`inf`|起點距離為 0|
|第 1 輪|`1` 條邊（0 次中轉）|`0`|`100`|`inf`|`inf`|邊 `0->1 (100)` 更新 `prices[1]`|
|第 2 輪|`2` 條邊（1 次中轉）|`0`|`100`|`200`|`700`|邊 `1->2 (100)` 更新 `prices[2]=200`<br><br>  <br><br>邊 `1->3 (600)` 更新 `prices[3]=700`|

最終返回 `prices[3] = 700`（若再多跑第 3 輪，邊 `2->3 (200)` 才會將 `prices[3]` 更新為 `400`，但受限於 `k = 1` 只能跑 2 輪）。

## **時間與空間複雜度分析**

設 $V$ 為城市數量（`n`），$E$ 為航班數量（`len(flights)`），$K$ 為最大中轉次數（`k`）。

- **時間複雜度：$O((V + E) \cdot K)$**
    
    - 外層迴圈執行 $K + 1$ 次。
        
    - 每次迴圈需要複製長度為 $V$ 的陣列（耗時 $O(V)$），並遍歷全部 $E$ 條航班進行鬆弛（耗時 $O(E)$）。
        
    - 總時間複雜度為 **$O((V + E) \cdot K)$**。
        
- **空間複雜度：$O(V)$**
    
    - 僅需維護 `prices` 與 `temp_prices` 兩個長度為 $V$ 的一維陣列，無需建立鄰接表，空間複雜度為 **$O(V)$**。
        

## **其他解法**

### **1. 限制層數的 BFS（隊列層序遍歷）**

- **思路**：
    
    - 建立鄰接表 `graph`，使用隊列 `queue` 儲存 `(當前城市, 累計花費)`，並維護一個全域陣列 `min_cost[i]` 記錄到達城市 `i` 的最低花費。
        
    - 每次處理當前層的所有節點（代表相同中轉次數），只有當 `curr_cost + price < min_cost[next_city]` 時才更新並入隊（避免重複無效擴展導致超時）。
        
    - 當層數 `stops > k` 時停止搜索。
        
- **時間複雜度**：$O(E \cdot K)$（最壞情況下每條邊在每一層被鬆弛一次）。
    
- **空間複雜度**：$O(V + E)$（需要額外儲存鄰接表與 BFS 隊列）。
    

### **2. 帶狀態記錄的 Dijkstra 算法（二維狀態堆）**

- **思路**：
    
    - 傳統 Dijkstra 的 `visited` 只記錄「城市 `u`」，但在本題中，**「花費較高但中轉次數更少」的路徑依然可能是合法解**，不能直接剪枝。
        
    - 需要將狀態擴展為二維：維護 `stops[u]` 記錄到達城市 `u` 時的**最少中轉次數**。當從最小堆彈出 `(cost, u, curr_stops)` 時，若 `curr_stops <= k` 且 `curr_stops < stops[u]`，才允許繼續向鄰居擴展。
        
- **時間複雜度**：$O(E \cdot K \log(E \cdot K))$。
    
- **空間複雜度**：$O(V \cdot K + E)$。
    

## **總結**

- **最推薦面試寫法**：**Bellman-Ford 算法（備份一維陣列版）**，代碼僅十幾行、無需建圖、不會出現 Dijkstra 狀態剪枝寫錯的 Bug，且空間複雜度僅 $O(V)$。
    
- **稀疏圖剪枝優化版**：**按層遍歷的 BFS + `min_cost` 剪枝**，在實際運行中通常比固定跑滿 `k + 1` 輪的 Bellman-Ford 更快結束。
    
- **核心思維關鍵**：只要看到題目出現「**最多經過 $K$ 步 / $K$ 次中轉的最短路徑**」，第一時間想到 **Bellman-Ford 限制迭代 $K+1$ 次** 或 **分層 BFS**。
    

## **圖論 / 帶限制最短路徑與動態規劃類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 787**|Cheapest Flights Within K Stops（K 站中轉內最便宜的航班）|**有向加權圖 + 邊數限制（最多 $K+1$ 條邊）最短路**|**Bellman-Ford（備份陣列） / 限制層數 BFS**|
|**LeetCode 743**|Network Delay Time（網絡延遲時間）|**有向加權圖單源最短路徑（無邊數限制）**|**標準 Dijkstra 算法（最小堆）**|
|**LeetCode 1928**|Minimum Cost to Reach Destination in Time（規定時間內到達終點的最小花費）|**有向加權圖 + 總時間上限約束最小費用**|**二維 DP（按時間步推進） / 二維狀態 Dijkstra**|
|**LeetCode 1514**|Path with Maximum Probability（概率最大的路徑）|**無向圖最大乘積路徑**|**Dijkstra 變體（最大堆，邊權相乘）**|
|**LeetCode 1334**|Find the City With the Smallest Number of Neighbors at a Threshold Distance|**多源最短路徑 + 距離閾值統計**|**Floyd-Warshall 算法 $O(V^3)$**|
|**LeetCode 1293**|Shortest Path in a Grid with Obstacles Elimination|**網格最短路 + 額外資源限制（最多消除 $K$ 個障礙）**|**BFS + 三維狀態訪問記錄 `(r, c, k)`**|
|**LeetCode 778**|Swim in Rising Water（水位上升的泳池中游泳）|**網格瓶頸最短路徑（路徑最大值最小）**|**Modified Dijkstra（最小堆） / 並查集**|

**進階變形題（LeetCode 1928 - Minimum Cost to Reach Destination in Time）：** 一個國家有 `n` 個城市，城市編號為 `0` 到 `n - 1` ，題目保證所有城市都由雙向道路連接在一起。道路由二維整數數組 `edges` 表示，其中 `edges[i] = [x_i, y_i, time_i]` 表示城市 `x_i` 和 `y_i` 之間有一條雙向道路，耗費時間為 `time_i` 分鐘。此外，每次經過一個城市 `i` 都需要繳納通行費 `passingFees[i]` 。現在你從城市 `0` 出發，希望在 `maxTime` 分鐘內（包含 `maxTime` 分鐘）到達城市 `n - 1` ，請你返回完成旅行的 **最小費用** ，如果無法在規定時間內到達，請返回 `-1` 。

Python

```
from typing import List

class Solution:
    def minCost(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        n = len(passingFees)
        # dp[t][i] 表示在恰好第 t 分钟到达城市 i 的最小通行费总和
        dp = [[float("inf")] * n for _ in range(maxTime + 1)]
        dp[0][0] = passingFees[0]
        
        for t in range(1, maxTime + 1):
            for u, v, travel_time in edges:
                if travel_time <= t:
                    if dp[t - travel_time][u] != float("inf"):
                        dp[t][v] = min(dp[t][v], dp[t - travel_time][u] + passingFees[v])
                    if dp[t - travel_time][v] != float("inf"):
                        dp[t][u] = min(dp[t][u], dp[t - travel_time][v] + passingFees[u])
                        
        ans = min(dp[t][n - 1] for t in range(maxTime + 1))
        return ans if ans != float("inf") else -1
```