
矩陣中的最長遞增路徑 Longest Increasing Path in a Matrix

LeetCode 329 / LintCode 200 给定一个 `m x n` 整数矩阵 `matrix` ，找出其中 **最长递增路径** 的长度。 对于每个单元格，你可以往上，下，左，右四个方向移动。 你 **不能** 在 **对角线** 方向上移动或移动到 **边界外**（即不允许环绕）。

**样例 1：** 输入：
```Python
matrix = [[9,9,4],[6,6,8],[2,1,1]]
```

输出：
```Python
4
```

解释： 最长递增路径为 [1, 2, 6, 9] 共 4 个单元格

**样例 2：** 输入：
```Python
matrix = [[3,4,5],[3,2,6],[2,2,1]]
```

输出：
```Python
4
```

解释： 最长递增路径是 [3, 4, 5, 6] 注意不允许在对角线方向上移动

**样例 3：** 输入：
```Python
matrix = [[1]]
```

输出：
```Python
1
```

解释： 单个单元格的最长递增路径长度为 1
```Python
# 解法一：拓撲排序 / Kahn's Algorithm (Bottom-Up DP / BFS)
from collections import deque

class Solution:
    def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        rows, cols = len(matrix), len(matrix[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        outdegree = [[0] * cols for _ in range(rows)]

        # 計算每個单元格的「出度」（有多少個鄰居的值嚴格大於自己）
        for r in range(rows):
            for c in range(cols):
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] > matrix[r][c]:
                        outdegree[r][c] += 1

        # 將所有出度為 0 的單元格（局部最大值 / 終點）加入佇列
        queue = deque()
        for r in range(rows):
            for c in range(cols):
                if outdegree[r][c] == 0:
                    queue.append((r, c))

        ans = 0
        # 逐層往數值更小的鄰居剝洋蔥，層數即為最長遞增路徑長度
        while queue:
            ans += 1
            for _ in range(len(queue)):
                r, c = queue.popleft()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] < matrix[r][c]:
                        outdegree[nr][nc] -= 1
                        if outdegree[nr][nc] == 0:
                            queue.append((nr, nc))

        return ans
```


```Python
# 解法二：記憶化搜索 (Top-Down DP / DFS + Memoization - 最優面試解)
class Solution:
    def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        rows, cols = len(matrix), len(matrix[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        # memo[r][c] 記錄從 (r, c) 出發能走出的最長遞增路徑長度
        memo = [[0] * cols for _ in range(rows)]

        def dfs(r: int, c: int) -> int:
            if memo[r][c] != 0:
                return memo[r][c]

            best = 1  # 至少包含自己這一個格子
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] > matrix[r][c]:
                    best = max(best, 1 + dfs(nr, nc))

            memo[r][c] = best
            return best

        ans = 0
        for r in range(rows):
            for c in range(cols):
                ans = max(ans, dfs(r, c))

        return ans
```

解釋: step1: 狀態定義 `memo[r][c]` 為從座標 `(r, c)` 出發所能走出的最長嚴格遞增路徑長度，初始全為 `0`（代表尚未計算）。 step2: 因為路徑必須「嚴格遞增」（`matrix[nr][nc] > matrix[r][c]`），圖中不可能存在環（DAG 有向無環圖），因此遞迴過程中不需要額外的 `visited` 陣列防回頭。 step3: 以样例 1 `matrix = [[9, 9, 4], [6, 6, 8], [2, 1, 1]]` 為例，從 `(0, 0)`（值為 `9`）開始呼叫 `dfs(0, 0)`。 step4: `(0, 0)` 四周鄰居為 `9` 和 `6`，皆不大於 `9` -> `memo[0][0] = 1`。同理 `(0, 1)`（值為 `9`）-> `memo[0][1] = 1`。 step5: 算到 `(1, 0)`（值為 `6`）時，上方鄰居 `(0, 0)` 為 `9` 大於 `6`，直接複用 `memo[0][0] = 1` -> `memo[1][0] = 1 + 1 = 2`。 step6: 算到 `(2, 0)`（值為 `2`）時，上方鄰居 `(1, 0)` 為 `6` 大於 `2`，複用 `memo[1][0] = 2` -> `memo[2][0] = 1 + 2 = 3`。 step7: 算到 `(2, 1)`（值為 `1`）時，左方鄰居 `(2, 0)` 為 `2` 大於 `1`，複用 `memo[2][0] = 3` -> `memo[2][1] = 1 + 3 = 4`。 step8: 遍歷所有格子取最大值 `max(memo)`，最終回傳 `4`（路徑 `[1, 2, 6, 9]`）。

解釋: 考慮從單元格 `(r, c)` 出發，下一步往上下左右哪個鄰居 `(nr, nc)` 走。 狀態轉移方程 `DP[r][c] = 1 + max(DP[nr][nc])`，其中 `(nr, nc)` 為滿足 `matrix[nr][nc] > matrix[r][c]` 的相鄰格子；若沒有任何鄰居比自己大，則 `DP[r][c] = 1`。 因為矩陣格子之間無法直接用普通雙層迴圈確定計算順序（數字大小散落在二維平面上），所以最適合使用 **DFS + 記憶化搜索**，讓遞迴自動先算完數值較大的後續格子。

### **LeetCode 329 / LintCode 200 - Longest Increasing Path in a Matrix（矩陣中的最長遞增路徑）**

這道題目是經典的 **二維記憶化搜索（2D Memoization / Top-Down DP）** 與 **有向無環圖（DAG）最長路徑** 問題。與一般網格 DP（只能向右或向下）不同，本題可以往 **四個方向** 移動，但「嚴格遞增」的限制天然消除了環路，使每個格子的最長路徑成為一個無後效性的子問題。

## **解法解析**

這道題可以轉化為 **DAG 上的動態規劃（記憶化搜索）** 來求解。

### **思路**

1. **觀察規律**
    
    - 題目要求路徑必須「嚴格遞增」，代表如果從格子 `A` 走到格子 `B`，必有 `val(A) < val(B)`，因此永遠不可能再走回 `A`。
        
    - 整個矩陣其實是一張 **有向無環圖（Directed Acyclic Graph, DAG）**，邊由數值小的格子指向相鄰數值大的格子。
        
    - 局部最大值的格子（四周鄰居都小於等於自己）沒有往外走的邊，它們的起點路徑長度固定為 `1`。
        
    - 其他格子的最長路徑，就是「所有比自己大的鄰居的最長路徑最大值」再加上 `1`。
        
2. **發現遞推公式**
    
    - `dfs(r, c) = 1 + max({ dfs(nr, nc) | (nr, nc) 為相鄰格子且 matrix[nr][nc] > matrix[r][c] })`
        
    - 初始條件（Base Case）：
        
        - 若 `(r, c)` 四周沒有比它大的鄰居，則 `dfs(r, c) = 1`
            
        - 使用二維陣列 `memo[r][c]` 快取計算結果（`0` 代表未訪問）
            
3. **代碼執行流程**
    
    - 建立與 `matrix` 同維度的 `memo` 陣列，全部初始化為 `0`。
        
    - 定義 `dfs(r, c)`：若 `memo[r][c] != 0`，直接回傳快取結果；否則檢查上下左右四個方向，找出所有數值嚴格大於 `matrix[r][c]` 的鄰居，遞迴取得它們的結果並取最大值加 `1`，存入 `memo[r][c]` 後回傳。
        
    - 主程式遍歷矩陣中的每一個格子 `(r, c)`，呼叫 `dfs(r, c)` 並持續更新全局最大值 `ans`。
        

## **具體舉例**

假設 `matrix = [[9, 9, 4], [6, 6, 8], [2, 1, 1]]`，各格子最終計算出的 `memo[r][c]` 數值對照如下：

|**座標 (r, c)**|**矩陣數值 matrix[r][c]**|**可轉移的較大鄰居**|**memo[r][c]（最長路徑長度）**|**對應最長路徑**|
|---|---|---|---|---|
|`(0, 0)`|`9`|無|`1`|`[9]`|
|`(0, 1)`|`9`|無|`1`|`[9]`|
|`(0, 2)`|`4`|`(0, 1)` 值 `9`、`(1, 2)` 值 `8`|`2`|`[4, 9]` 或 `[4, 8]`|
|`(1, 0)`|`6`|`(0, 0)` 值 `9`|`2`|`[6, 9]`|
|`(1, 1)`|`6`|`(0, 1)` 值 `9`、`(1, 2)` 值 `8`|`2`|`[6, 9]` 或 `[6, 8]`|
|`(1, 2)`|`8`|無|`1`|`[8]`|
|`(2, 0)`|`2`|`(1, 0)` 值 `6`|`3`|`[2, 6, 9]`|
|`(2, 1)`|`1`|`(2, 0)` 值 `2`、`(1, 1)` 值 `6`|`4`|`[1, 2, 6, 9]`|
|`(2, 2)`|`1`|`(1, 2)` 值 `8`|`2`|`[1, 8]`|

遍歷 `memo` 陣列後取最大值，得到 `(2, 1)` 處的 `4`，代表最長遞增路徑長度為 **4**。

## **時間與空間複雜度分析**

- **時間複雜度：O(m × n)**（其中 `m` 為列數，`n` 為行數）
    
    - 共有 `m × n` 個狀態，透過 `memo` 陣列快取後，每個格子最多只會被真正計算一次；每次計算檢查 `4` 個方向，均攤為 `O(1)`，因此總時間複雜度為 **O(m × n)**。
        
- **空間複雜度：O(m × n)**
    
    - `memo` 陣列需要 `O(m × n)` 空間，且最壞情況下（矩陣呈蛇形單調遞增）DFS 遞迴呼叫棧深度可達 `O(m × n)`，因此總空間複雜度為 **O(m × n)**。
        

## **其他解法**

### **1. 拓撲排序 / 剝洋蔥 BFS（Topological Sort / Kahn's Algorithm）**

- **思路**：
    
    - 將每個單元格視為圖的節點，若 `matrix[nr][nc] > matrix[r][c]`，則連一條有向邊，統計每個節點的 **出度（outdegree）**。
        
    - 把所有出度為 `0` 的局部最大值放入 BFS 佇列，每一輪將當前層全部彈出並使相鄰較小節點的出度減 `1`，當某節點出度變為 `0` 時入隊。**BFS 總共擴展的層數**即為最長遞增路徑長度。
        
- **時間複雜度**：O(m × n)
    
- **空間複雜度**：O(m × n)（`outdegree` 陣列與 BFS 佇列，無遞迴棧溢出風險）
    

### **2. 排序 + 動態規劃（Sorting + Bottom-Up DP）**

- **思路**：
    
    - 將所有單元格按數值從大到小（或從小到大）排序，存入陣列 `cells`。
        
    - 依序遍歷排序後的格子，直接用一維/二維 `dp` 陣列轉移 `dp[r][c] = 1 + max(dp[nr][nc])`。因為按數值大小排過序，計算時所有較大鄰居的 `dp` 值必定已經算好。
        
- **時間複雜度**：O(m × n × log(m × n))（瓶頸在排序）
    
- **空間複雜度**：O(m × n)
    

## **總結**

- **最直觀的解法**：**排序 + Bottom-Up DP**，強制把二維無序格點轉成一維線性順序，時間 `O(mn log(mn))`，空間 `O(mn)`。
    
- **最佳解法（面試首選）**：**DFS + 記憶化搜索（Top-Down DP）**，代碼最簡潔直觀，時間 `O(m × n)`，空間 `O(m × n)`。
    
- **工程防爆棧解法**：當矩陣非常大（如 `1000 × 1000`）且呈蛇形遞增時，Python 預設遞迴深度（`1000`）會觸發 `RecursionError`，此時改用 **拓撲排序 BFS（Kahn's Algorithm）** 最安全。
    

## **LeetCode / LintCode 網格路徑與記憶化搜索類問題整理**

|**题号**|**题目名称**|**问题类型**|**解法**|
|---|---|---|---|
|**LeetCode 329 / LintCode 200**|Longest Increasing Path in a Matrix（矩阵中的最长递增路径）|**2D 网格四方向严格递增，求最长路径**|**DFS + 记忆化搜索 `O(mn)`** 或 **拓扑排序 BFS**|
|**LeetCode 62 / LintCode 114**|Unique Paths（不同路径）|**2D 网格仅向右或向下，求路径总数**|**动态规划 `dp[i][j] = dp[i-1][j] + dp[i][j-1]`**|
|**LeetCode 63 / LintCode 115**|Unique Paths II（不同路径 II）|**2D 网格带障碍物，求路径总数**|**动态规划，遇到障碍物时 `dp[i][j] = 0`**|
|**LeetCode 64 / LintCode 110**|Minimum Path Sum（最小路径和）|**2D 网格带非负权重，求最小路径和**|**动态规划 `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`**|
|**LeetCode 2328**|Number of Increasing Paths in a Grid（网格图中递增路径的数目）|**2D 网格四方向严格递增，求所有递增路径总数**|**DFS + 记忆化搜索 `dp[r][c] = 1 + sum(dp[nr][nc])`**|
|**LeetCode 300 / LintCode 76**|Longest Increasing Subsequence（最长递增子序列）|**1D 数组求最长严格递增子序列（LIS）**|**动态规划 `O(n²)`** 或 **贪心 + 二分查找 `O(n log n)`**|
|**LeetCode 673**|Number of Longest Increasing Subsequence（最长递增子序列的个数）|**1D 数组求最长递增子序列的方案数**|**双状态 DP（同时维护 `length[i]` 与 `count[i]`）**|
|**LeetCode 207 / LintCode 615**|Course Schedule（课程表）|**有向图判环 / 拓扑排序**|**Kahn's Algorithm（入度表 + BFS）** 或 **三色标记 DFS**|

### **延伸變形題：網格圖中遞增路徑的數目（LeetCode 2328）- 求「路徑總數」而非「最長長度」**

给你一个 `m x n` 的整数网格图 `grid` ，你可以从一个格子移动到 `4` 个方向相邻的任意一个格子。请你返回在网格图中从 **任意** 格子出发，达到 **任意** 格子，且路径中的数字是 **严格递增** 的路径数目。由于答案可能会很大，请将结果对 `10^9 + 7` 取余后返回。

Python

```
class Solution:
    def countPaths(self, grid: list[list[int]]) -> int:
        MOD = 10**9 + 7
        rows, cols = len(grid), len(grid[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        # memo[r][c] 记录从 (r, c) 出发的严格递增路径总数
        memo = [[0] * cols for _ in range(rows)]

        def dfs(r: int, c: int) -> int:
            if memo[r][c] != 0:
                return memo[r][c]

            # 关键点：自身单独构成一条长度为 1 的路径，因此初始 count = 1
            count = 1
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > grid[r][c]:
                    # 将 max 改为累加，即可从「求最长长度」变为「求路径总数」
                    count = (count + dfs(nr, nc)) % MOD

            memo[r][c] = count
            return count

        total = 0
        for r in range(rows):
            for c in range(cols):
                total = (total + dfs(r, c)) % MOD

        return total
```