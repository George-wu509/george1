
連接所有點的最小費用 Min Cost to Connect All Points

Leetcode 1584 给你一个 `points` 数组，表示 2D 平面上的一些点，其中 `points[i] = [x_i, y_i]` 。

连接点 `[x_i, y_i]` 和点 `[x_j, y_j]` 的费用为它们之间的 **曼哈顿距离（Manhattan Distance）**：`|x_i - x_j| + |y_i - y_j|` ，其中 `|val|` 表示 `val` 的绝对值。

请你返回将所有点连接的最小总费用。只有任意两点之间 **有且仅有** 一条简单路径时，才认为所有点都已连接。

**样例 1：** 输入：
```Python
points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
```

输出：
```Python
20
```

解释： 连接方式如下：

- `[0,0]` 与 `[2,2]` 费用为 `4`
- `[2,2]` 与 `[5,2]` 费用为 `3`
- `[5,2]` 与 `[7,0]` 费用为 `4`
- `[2,2]` 与 `[3,10]` 费用为 `9` 总费用为 `4 + 3 + 4 + 9 = 20` ，所有点连通且无环。

**样例 2：** 输入：
```Python
points = [[3,12],[-2,5],[-4,1]]
```

输出：
```Python
18
```

解释：

- `[-4,1]` 到 `[-2,5]` 距离为 `|-4 - (-2)| + |1 - 5| = 2 + 4 = 6`
- `[-2,5]` 到 `[3,12]` 距离为 `|-2 - 3| + |5 - 12| = 5 + 7 = 12` 总费用为 `6 + 12 = 18` 。

**样例 3（只有一个点的边界情况）：** 输入：
```Python
points = [[0,0]]
```

输出：
```Python
0
```

解释： 只有一个点，不需要任何边即可连通，总费用为 `0` 。

```Python
from typing import List

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        # 针对稠密图的 Prim 算法优化版：O(N^2) 时间，O(N) 空间
        n = len(points)
        min_dist = [float("inf")] * n  # 记录每个未加入点到当前生成树的最小距离
        min_dist[0] = 0
        visited = [False] * n
        total_cost = 0
        
        for _ in range(n):
            # 1. 在所有未访问节点中，找出距离当前生成树最近的节点 curr
            curr = -1
            for i in range(n):
                if not visited[i] and (curr == -1 or min_dist[i] < min_dist[curr]):
                    curr = i
                    
            # 2. 将 curr 加入最小生成树，累加费用
            visited[curr] = True
            total_cost += min_dist[curr]
            
            # 3. 用新加入的 curr 更新其余未访问节点到生成树的最小距离
            x1, y1 = points[curr]
            for nxt in range(n):
                if not visited[nxt]:
                    x2, y2 = points[nxt]
                    dist = abs(x1 - x2) + abs(y1 - y2)
                    if dist < min_dist[nxt]:
                        min_dist[nxt] = dist
                        
        return total_cost
```


```Python
from typing import List

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        # Kruskal 算法 + 並查集（Union-Find）
        n = len(points)
        edges = []
        
        # 1. 枚举所有点对，构建完全图的全部边
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                edges.append((dist, i, j))
                
        # 2. 按边权（曼哈顿距离）从小到大排序
        edges.sort()
        
        # 3. 并查集初始化
        parent = list(range(n))
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
            
        total_cost = 0
        edges_used = 0
        
        # 4. 贪心选取最小边，若不构成环则合并
        for dist, u, v in edges:
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                parent[root_u] = root_v
                total_cost += dist
                edges_used += 1
                if edges_used == n - 1:
                    break
                    
        return total_cost
```

解釋: step1: 題目要求用總權重最小的邊將 `n` 個點連成一棵樹（任意兩點有且僅有一條路徑），這是標準的 **無向完全圖最小生成樹（Minimum Spanning Tree, MST）** 問題。 step2: 以 `points = [[0,0], [2,2], [3,10], [5,2], [7,0]]`（索引 `0~4`）為例，使用陣列版 Prim 算法，初始化 `min_dist = [0, inf, inf, inf, inf]`，`visited = [False] * 5`，`total_cost = 0`。 step3: 第 1 輪：選出未訪問中 `min_dist` 最小的點 `0`（距離 `0`），標記 `visited[0] = True`，`total_cost += 0`。用點 `0` 更新其餘點距離 -> `min_dist = [0, 4, 13, 7, 7]`。 step4: 第 2 輪：選出未訪問中 `min_dist` 最小的點 `1`（距離 `4`），標記 `visited[1] = True`，`total_cost += 4`（累計 `4`）。用點 `1` 更新其餘未訪問點（`2, 3, 4`） -> 點 `2` 變 `9`，點 `3` 變 `3`，點 `4` 保持 `7` -> `min_dist = [0, 4, 9, 3, 7]`。 step5: 第 3 輪：選出未訪問中 `min_dist` 最小的點 `3`（距離 `3`），標記 `visited[3] = True`，`total_cost += 3`（累計 `7`）。用點 `3` 更新點 `2, 4` -> 點 `4` 距離從 `7` 縮短為 `4` -> `min_dist = [0, 4, 9, 3, 4]`。 step6: 第 4 輪：選出未訪問中 `min_dist` 最小的點 `4`（距離 `4`），標記 `visited[4] = True`，`total_cost += 4`（累計 `11`）。點 `2` 距離仍為 `9`。 step7: 第 5 輪：選出最後一個未訪問點 `2`（距離 `9`），標記 `visited[2] = True`，`total_cost += 9`（累計 `20`），所有點納入生成樹，返回 `20`。

解釋: 任意兩點間都存在一條帶權邊，因此這是一張有 $N$ 個頂點、$\frac{N(N-1)}{2}$ 條邊的**無向完全圖（稠密圖）**。 在稠密圖中，邊數 $E \approx N^2$：

- **Kruskal 算法** 需要先存下所有 $O(N^2)$ 條邊並排序，時間複雜度為 $O(N^2 \log N)$，空間複雜度為 $O(N^2)$。
    
- **無堆優化的 Prim 算法** 每次直接掃描一維陣列找最近點，無需預先建立邊集，時間複雜度嚴格為 **$O(N^2)$**，空間複雜度僅為 **$O(N)$**，是本題的最優解。
    

### **LeetCode 1584 - Min Cost to Connect All Points（連接所有點的最小費用）**

這道題目是圖論中非常經典的 **最小生成樹（Minimum Spanning Tree, MST）** 模板題。題目給定平面上的 $N$ 個點，邊權為兩點間的曼哈頓距離，要求用最小總邊權連通所有點，標準解法為 **Prim 算法（稠密圖陣列版）** 與 **Kruskal 算法 + 並查集（Union-Find）**。

## **解法解析**

### **思路**

1. **問題轉化（圖論建模）**
    
    - **節點（Vertex）**：平面上的 $N$ 個點，編號為 `0` 到 `N - 1`。
        
    - **無向邊（Undirected Edge）**：任意兩點 `i` 和 `j` 之間都有一條無向邊，權重為曼哈頓距離 `|x_i - x_j| + |y_i - y_j|`。
        
    - **目標**：「任意兩點之間有且僅有一條簡單路徑」是**樹（Tree）**的嚴格定義，因此題目即求這張完全圖的**最小生成樹（MST）權重和**。
        
2. **Prim 算法核心思想（加點法）**
    
    - 將所有點分為兩類：**已加入生成樹的點集 $S$** 與 **尚未加入的點集 $V \setminus S$**。
        
    - 維護一個一維陣列 `min_dist[i]`，表示未加入點 `i` 到**整個點集 $S$ 中所有點的最短距離**。
        
    - 每一輪執行兩個動作：
        
        1. **貪心選點**：在所有未加入的點中，挑選 `min_dist` 最小的那個點 `curr`，將其納入 $S$ 並把 `min_dist[curr]` 計入總費用。
            
        2. **鬆弛更新**：因為 `curr` 剛加入 $S$，它可能成為某些未加入點 `nxt` 通往 $S$ 的更近橋樑，因此遍歷所有未加入點並更新 `min_dist[nxt] = min(min_dist[nxt], dist(curr, nxt))`。
            
3. **代碼執行流程**
    
    - 初始化 `min_dist = [0, inf, ..., inf]`（默認從 `0` 號點開始生長），`visited = [False] * n`。
        
    - 執行 `n` 次迴圈，每次找未訪問且 `min_dist` 最小的節點 `curr`。
        
    - 標記 `visited[curr] = True`，累加 `total_cost += min_dist[curr]`。
        
    - 遍歷其餘未訪問節點 `nxt`，即時計算曼哈頓距離並更新 `min_dist[nxt]`。
        

## **具體舉例**

假設 `points = [[0,0], [2,2], [3,10], [5,2], [7,0]]`（節點 `0 ~ 4`），Prim 算法執行過程如下：

|**輪次**|**選中最近點 curr**|**納入費用 min_dist[curr]**|**visited 狀態**|**更新後的 min_dist（未訪問點）**|**累計 total_cost**|
|---|---|---|---|---|---|
|初始化|-|-|`[F, F, F, F, F]`|`[0, inf, inf, inf, inf]`|`0`|
|第 1 輪|`0` `[0,0]`|`0`|`[T, F, F, F, F]`|`[-, 4, 13, 7, 7]`|`0`|
|第 2 輪|`1` `[2,2]`|`4`|`[T, T, F, F, F]`|`[-, -, 9, 3, 7]`|`4`|
|第 3 輪|`3` `[5,2]`|`3`|`[T, T, F, T, F]`|`[-, -, 9, -, 4]`|`7`|
|第 4 輪|`4` `[7,0]`|`4`|`[T, T, F, T, T]`|`[-, -, 9, -, -]`|`11`|
|第 5 輪|`2` `[3,10]`|`9`|`[T, T, T, T, T]`|`[-, -, -, -, -]`|`20`|

最終返回 `total_cost = 20`。

## **時間與空間複雜度分析**

設 $N$ 為點的數量（`len(points)`），圖中總邊數 $E = \frac{N(N-1)}{2} = O(N^2)$。

- **時間複雜度：$O(N^2)$**
    
    - 外層迴圈執行 $N$ 次，每次將一個頂點加入生成樹。
        
    - 內層包含兩個平行的 $O(N)$ 迴圈：一個用於尋找最小 `min_dist` 頂點，另一個用於更新剩餘未訪問頂點的距離。
        
    - 總時間複雜度為 **$O(N^2)$**，在完全圖上優於需要排序所有邊的 $O(N^2 \log N)$。
        
- **空間複雜度：$O(N)$**
    
    - 僅需維護長度為 $N$ 的 `min_dist` 與 `visited` 陣列，無需額外儲存 $O(N^2)$ 條邊，空間複雜度為 **$O(N)$**。
        

## **其他解法**

### **1. Kruskal 算法 + 並查集（加邊法）**

- **思路**：
    
    - 計算所有點對之間的曼哈頓距離，生成全部 $\frac{N(N-1)}{2}$ 條邊 `(dist, u, v)`。
        
    - 將所有邊按權重從小到大排序。
        
    - 依序遍歷每條邊，利用並查集（Union-Find）判斷 `u` 和 `v` 是否已在同一連通分量中；若不連通，則合併兩者並累加權重，直到選滿 `N - 1` 條邊為止。
        
- **時間複雜度**：$O(N^2 \log N)$（瓶頸在於對 $O(N^2)$ 條邊進行排序）。
    
- **空間複雜度**：$O(N^2)$（需要列表儲存所有邊）。
    

### **2. 最小堆優化的 Prim 算法（Heap-based Prim）**

- **思路**：
    
    - 使用最小堆 `min_heap` 儲存 `(dist, node)`，每次彈出堆頂最小距離的未訪問節點，再將其鄰邊推入堆中。
        
- **特點與對比**：
    
    - 堆優化 Prim 的時間複雜度為 $O(E \log V) = O(N^2 \log N)$，空間複雜度為 $O(N^2)$。
        
    - 堆優化版本僅適合**稀疏圖**（$E \ll V^2$）；在像本題這樣的**稠密完全圖**中，直接用陣列遍歷找最小值的**樸素 Prim 算法（$O(N^2)$ 時間、$O(N)$ 空間）反而更快更省內存**。
        

## **總結**

- **本題最佳解法**：**樸素 Prim 算法（一維陣列版）**，時間複雜度 $O(N^2)$，空間複雜度僅 $O(N)$，完美契合完全圖特性。
    
- **最通用面試模板**：**Kruskal 算法 + 並查集**，當題目直接給定邊列表 `edges`（尤其是稀疏圖）時，Kruskal 寫法最直觀、最不易出錯。
    
- **選型口訣**：**「稀疏圖用 Kruskal 或堆優化 Prim（$O(E \log E)$）；稠密圖 / 完全圖用陣列版 Prim（$O(V^2)$）」**。
    

## **圖論 / 最小生成樹（MST）與並查集類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 1584**|Min Cost to Connect All Points（連接所有點的最小費用）|**無向完全圖最小生成樹（曼哈頓距離）**|**Prim 算法（$O(N^2)$ 陣列版） / Kruskal + 並查集**|
|**LeetCode 1135**|Connecting Cities With Minimum Cost（最低成本聯通所有城市）|**無向稀疏圖最小生成樹（判斷是否能全連通）**|**Kruskal 算法 + 並查集（檢查最終邊數是否為 $N-1$）**|
|**LeetCode 1168**|Optimize Water Distribution in a Village（水資源分配優化）|**帶點權（打井費用）與邊權（管道費用）的 MST**|**建立虛擬超級源點 `0` 連向各點（邊權=打井費）+ Kruskal**|
|**LeetCode 1489**|Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree|**最小生成樹的關鍵邊與偽關鍵邊判定**|**Kruskal + 枚舉刪邊 / 強制選邊對比 MST 權重**|
|**LeetCode 684**|Redundant Connection（冗餘連接）|**無向圖找多餘的一條邊（判環）**|**並查集（Union-Find）**|
|**LeetCode 778**|Swim in Rising Water（水位上升的泳池中游泳）|**網格瓶頸最短路徑（最小生成樹等價問題）**|**Modified Dijkstra / 並查集按高度合併**|
|**LeetCode 1631**|Path With Minimum Effort（最小體力消耗路徑）|**網格瓶頸最短路徑（相鄰差值最大最小）**|**Modified Dijkstra / 並查集（Kruskal 思想）**|

**進階變形題（LeetCode 1168 - Optimize Water Distribution in a Village）：** 村裡面一共有 `n` 棟房子。我們希望通過建造水井和鋪設管道來為所有房子供水。對於每個房子 `i`（從 `1` 到 `n`），我們有兩種選擇：一種是直接在房子內建造水井，成本為 `wells[i - 1]`；另一種是從另一口井鋪設管道引水過來，成本由 `pipes` 給出，其中 `pipes[j] = [house1_j, house2_j, cost_j]` 代表用管道連接兩棟房子的雙向成本。請你返回為所有房子都供水的 **最低總成本** 。

Python

```
from typing import List

class Solution:
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        # 核心技巧：引入虚拟超级水源节点 0，在房子 i 打井等价于连接边 (0, i, wells[i-1])
        edges = []
        for i, cost in enumerate(wells):
            edges.append((cost, 0, i + 1))
            
        for u, v, cost in pipes:
            edges.append((cost, u, v))
            
        # 对所有 n + 1 个节点（0 到 n）运行标准 Kruskal 算法
        edges.sort()
        parent = list(range(n + 1))
        
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
            
        total_cost = 0
        edges_used = 0
        
        for cost, u, v in edges:
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                parent[root_u] = root_v
                total_cost += cost
                edges_used += 1
                # 总共有 n + 1 个节点，需要 n 条边连通
                if edges_used == n:
                    break
                    
        return total_cost
```