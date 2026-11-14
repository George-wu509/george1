
LeetCode 417 / LintCode 778 (NeetCode 150)
有一個 `m x n` 的矩形島嶼，與 **太平洋** 和 **大西洋** 相鄰。 **「太平洋」** 處於大陸的左邊界和上邊界，而 **「大西洋」** 處於大陸的右邊界和下邊界。
這個島被分割成一個由若干方形單元格組成的網格。給定一個 `m x n` 的整數矩陣 `heights` ， `heights[r][c]` 表示座標 `(r, c)` 上單元格 **高於海平面的高度** 。

島上雨水較多，如果相鄰單元格的高度 **小於或等於** 當前單元格的高度，雨水可以直接向北、南、東、西流向相鄰單元格。水可以從海洋附近的任何單元格流入海洋。

返回網格座標 `result` 的 **2D 列表** ，其中 `result[i] = [ri, ci]` 表示雨水從單元格 `(ri, ci)` 流動 **既可流向太平洋也可流向大西洋** 。

**樣例 1：**
輸入：
```python
heights = [
  [1, 2, 2, 3, 5],
  [3, 2, 3, 4, 4],
  [2, 4, 5, 3, 1],
  [6, 7, 1, 4, 5],
  [5, 1, 1, 2, 4]
]
```
輸出：
```python
[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]]
```
解釋：
以下座標的雨水既能流入太平洋（上/左邊界），也能流入大西洋（下/右邊界）：
- `[0, 4]`：本身同時位於上邊界（太平洋）與右邊界（大西洋）。
- `[1, 3]`：`[1,3] -> [0,3]`（太平洋）；`[1,3] -> [1,4]`（大西洋）。
- `[1, 4]`：`[1,4] -> [1,3] -> [0,3]`（太平洋）；本身位於右邊界（大西洋）。
- `[2, 2]`：`[2,2] -> [1,2] -> [0,2]`（太平洋）；`[2,2] -> [2,3] -> [2,4]`（大西洋）。
- `[3, 0]`、`[3, 1]`、`[4, 0]` 同理可通往兩大洋。

**樣例 2：**
輸入：
```python
heights = [[1]]
```
輸出：
```python
[[0, 0]]
```
解釋：
唯一的一格同時與太平洋和大西洋相鄰，雨水可以直接流入兩大洋

**樣例 3：**
輸入：
```python
heights = [
  [2, 1],
  [1, 2]
]
```
輸出：
```python
[[0, 0], [0, 1], [1, 0], [1, 1]]
```
解釋：
- `(0, 0)` 與 `(1, 1)` 同時接觸太平洋與大西洋邊界。
- `(0, 1)` 可直接流入上邊界（太平洋）與右邊界（大西洋）。
- `(1, 0)` 可直接流入左邊界（太平洋）與下邊界（大西洋）。

---

### 解法一：雙邊界逆向廣度優先搜尋（Boundary Multi-Source BFS，避免大型矩陣遞迴爆棧）

```python
from collections import deque

class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []

        ROWS, COLS = len(heights), len(heights[0])
        pac_queue, atl_queue = deque(), deque()
        pac_visited, atl_visited = set(), set()

        # 1. 初始化兩大洋的邊界起點：上/左邊界屬於太平洋，下/右邊界屬於大西洋
        for r in range(ROWS):
            pac_queue.append((r, 0))
            pac_visited.add((r, 0))
            atl_queue.append((r, COLS - 1))
            atl_visited.add((r, COLS - 1))

        for c in range(COLS):
            pac_queue.append((0, c))
            pac_visited.add((0, c))
            atl_queue.append((ROWS - 1, c))
            atl_visited.add((ROWS - 1, c))

        # 2. 定義逆向 BFS 函式：從海岸線往「高度大於等於當前格子」的高處爬升
        def bfs(queue: deque, visited: set) -> None:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            while queue:
                r, c = queue.popleft()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < ROWS and 0 <= nc < COLS and
                        (nr, nc) not in visited and
                        heights[nr][nc] >= heights[r][c]
                    ):
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        bfs(pac_queue, pac_visited)
        bfs(atl_queue, atl_visited)

        # 3. 取兩個集合的交集，即為同時可流向兩大洋的座標
        return [[r, c] for r in range(ROWS) for c in range(COLS) if (r, c) in pac_visited and (r, c) in atl_visited]
```
pass

### 解法二：雙邊界逆向深度優先搜尋（NeetCode 經典最佳解，程式碼最精簡）

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        ROWS, COLS = len(heights), len(heights[0])
        pac, atl = set(), set()

        def dfs(r: int, c: int, visit: set, prevHeight: int) -> None:
            # 越界、已經走訪過，或當前格子高度「低於」前一個格子（水無法從高處流下來），直接返回
            if (
                (r, c) in visit or
                r < 0 or c < 0 or
                r == ROWS or c == COLS or
                heights[r][c] < prevHeight
            ):
                return

            visit.add((r, c))
            dfs(r + 1, c, visit, heights[r][c])
            dfs(r - 1, c, visit, heights[r][c])
            dfs(r, c + 1, visit, heights[r][c])
            dfs(r, c - 1, visit, heights[r][c])

        # 1. 從上邊界（太平洋）與下邊界（大西洋）所有格子啟動逆向 DFS
        for c in range(COLS):
            dfs(0, c, pac, heights[0][c])
            dfs(ROWS - 1, c, atl, heights[ROWS - 1][c])

        # 2. 從左邊界（太平洋）與右邊界（大西洋）所有格子啟動逆向 DFS
        for r in range(ROWS):
            dfs(r, 0, pac, heights[r][0])
            dfs(r, COLS - 1, atl, heights[r][COLS - 1])

        # 3. 收集同時存在於 pac 與 atl 集合中的座標
        res = []
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) in pac and (r, c) in atl:
                    res.append([r, c])
        return res
```

解釋:
step1: 核心逆向思維（水往低處流 -> 逆向往高處爬）：如果從島上每個格子出發往下流，每個點都要跑一次搜尋，會有大量重複路徑；但如果反過來，分別從 **太平洋海岸線（上、左邊界）** 和 **大西洋海岸線（下、右邊界）** 出發，只往 **高度大於等於當前格子（`heights[nr][nc] >= heights[r][c]`）** 的方向逆流而上，就能一次找出所有能流進該海洋的格子！
step2: 假設 `heights = [[2, 1], [1, 2]]`（`2 x 2` 網格）。
step3: **太平洋逆向搜尋（從上邊界 `r = 0` 與左邊界 `c = 0` 出發）**：
  - `(0, 0)`（高度 `2`）入隊並加入 `pac`：相鄰 `(0, 1)` 高度 `1 < 2` 無法爬升，`(1, 0)` 高度 `1 < 2` 無法爬升。
  - `(0, 1)`（高度 `1`）本身就在上邊界，加入 `pac`：相鄰 `(1, 1)` 高度 `2 >= 1`（水可以從 `(1,1)` 流到 `(0,1)` 再入海！），因此 `(1, 1)` 也加入 `pac`。
  - `(1, 0)`（高度 `1`）本身就在左邊界，加入 `pac`。
  - 太平洋可達集合 `pac = {(0, 0), (0, 1), (1, 0), (1, 1)}`。
step4: **大西洋逆向搜尋（從下邊界 `r = 1` 與右邊界 `c = 1` 出發）**：
  - `(1, 1)`（高度 `2`）入隊並加入 `atl`：相鄰高度皆為 `1 < 2` 無法爬升。
  - `(0, 1)`（高度 `1`）位於右邊界，加入 `atl`：相鄰 `(0, 0)` 高度 `2 >= 1`，因此 `(0, 0)` 也加入 `atl`。
  - `(1, 0)`（高度 `1`）位於下邊界，加入 `atl`。
  - 大西洋可達集合 `atl = {(1, 1), (0, 1), (1, 0), (0, 0)}`。
step5: **取兩集合交集 `pac & atl`**：四個座標都在兩個集合內，最終回傳 `[[0, 0], [0, 1], [1, 0], [1, 1]]`。

---

### **LeetCode 417 - Pacific Atlantic Water Flow（太平洋大西洋水流問題）**

這道題目是二維網格圖論中 **「雙邊界逆向搜尋（Reverse Multi-Source DFS / BFS）」** 的最高頻考題之一，與 LeetCode 130（被圍繞的區域）屬於同一套核心設計模式。

---

## **解法解析**

這道題的難點不在於圖論演算法本身，而在於 **搜尋方向的翻轉** 與 **水流條件的反轉**。

### **思路**

1. **為什麼不能從內部格子正面往下流？**
   - 如果對每個格子 `(r, c)` 都啟動一次 DFS/BFS 往下流，我們必須同時追蹤「是否碰到上/左邊界」與「是否碰到下/右邊界」，而且在路徑分支時，單純的 `visited` 會阻擋其他等高路徑，最壞情況時間複雜度會飆升至 **O((m × n)²)**。
2. **逆向思維：從兩大洋分別「逆流爬山」**
   - 水流規則是：`新格子高度 <= 當前格子高度`（水往低處或平地流）。
   - 當我們改從海岸線（邊界）出發逆向往內陸走時，條件必須反轉為：**`新格子高度 >= 當前格子高度`**（只有比海岸更高的內陸，雨水才流得下來）。
   - 我們維護兩個獨立的訪問集合 `pac` 與 `atl`：
     - 跑一次以 **上邊界 + 左邊界** 為起點的 DFS/BFS，把所有能走到的格子記錄在 `pac` 中；
     - 再跑一次以 **下邊界 + 右邊界** 為起點的 DFS/BFS，把所有能走到的格子記錄在 `atl` 中。
3. **交集即為答案**
   - 只要某個座標 `(r, c)` 同時存在於 `pac` 和 `atl` 中，就代表從 `(r, c)` 出發的雨水既有一條非遞增路徑通往太平洋，也有一條非遞增路徑通往大西洋。

---

## **具體舉例**

以樣例 1 的 `5 x 5` 矩陣為例，觀察 `pac` 與 `atl` 各自擴散覆蓋的網格區域（`P` 表示可達太平洋，`A` 表示可達大西洋，`PA` 表示兩者皆可達）：

| 列索引 | 高度矩陣 `heights` | 太平洋可達 `pac` | 大西洋可達 `atl` | 兩者交集 `pac & atl`（最終答案） |
|---|---|---|---|---|
| **Row 0** | `[1, 2, 2, 3, 5]` | `P  P  P  P  P` | `.  .  .  .  A` | `[0, 4]` |
| **Row 1** | `[3, 2, 3, 4, 4]` | `P  P  P  P  P` | `.  .  .  A  A` | `[1, 3], [1, 4]` |
| **Row 2** | `[2, 4, 5, 3, 1]` | `P  P  P  .  .` | `.  .  A  A  A` | `[2, 2]` |
| **Row 3** | `[6, 7, 1, 4, 5]` | `P  P  .  .  .` | `A  A  A  A  A` | `[3, 0], [3, 1]` |
| **Row 4** | `[5, 1, 1, 2, 4]` | `P  .  .  .  .` | `A  A  A  A  A` | `[4, 0]` |

- **重點觀察 `(1, 4)`（高度 `4`）**：雖然它不直接接觸太平洋邊界，但它可以往左流向 `(1, 3)`（高度 `4` 等高流動），再往上流向 `(0, 3)`（高度 `3` 流入太平洋）；同時它本身就在右邊界（直接流入大西洋），因此 `[1, 4]` 位於交集中。

---

## **時間與空間複雜度分析**

設矩陣大小為 `M × N`（`M` 列 `N` 行）。

- **時間複雜度：O(M × N)**
  - 太平洋的逆向 DFS / BFS 中，每個格子最多只會被加入 `pac` 集合並訪問一次，耗時 `O(M × N)`。
  - 大西洋的逆向 DFS / BFS 中，每個格子同樣最多只會被加入 `atl` 集合並訪問一次，耗時 `O(M × N)`。
  - 最後遍歷 `M × N` 個座標檢查交集，耗時 `O(M × N)`。
  - 整體時間複雜度為線性 **O(M × N)**。
- **空間複雜度：O(M × N)**
  - 需要兩個集合 `pac` 與 `atl` 各自儲存最多 `M × N` 個座標，佔用 `O(M × N)` 空間。
  - DFS 的遞迴呼叫堆疊（或 BFS 的 `queue`）在最壞情況下（例如所有格子高度相同呈蛇形盤繞）深度/長度可達 `O(M × N)`。
  - 整體空間複雜度為 **O(M × N)**。

---

## **其他解法**

### **1. 位元遮罩狀態壓縮優化（Bitmask Grid Optimization）**
- **思路**：
  - 不使用兩個 Python `set()` 儲存 `(r, c)` tuple（減少雜湊與物件記憶體開銷），而是建立一個 `M × N` 的整數二維陣列 `state`：
    - 用二進位第 `0` 位（`1`，即 `0b01`）表示可到達太平洋；
    - 用二進位第 `1` 位（`2`，即 `0b10`）表示可到達大西洋。
  - 跑太平洋搜尋時對格子執行 `state[r][c] |= 1`，跑大西洋搜尋時執行 `state[r][c] |= 2`。最後只要收集所有 `state[r][c] == 3`（即 `0b11`）的座標即可。
- **時間複雜度**：O(M × N)（純陣列索引與位元運算，實測常數時間比 `set` 更快）
- **空間複雜度**：O(M × N)

---

## **總結**

- **面試最佳解**：**雙邊界逆向 DFS（解法二）** 或 **雙邊界逆向 BFS（解法一）**，時間 `O(M × N)`，空間 `O(M × N)`。面試時優先寫出程式碼最簡潔的雙邊界逆向 DFS，並主動向面試官補充「若矩陣規模極大可改用 BFS 避免遞迴深度限制」以及「`heights[nr][nc] >= heights[r][c]` 的等號邊界條件」。
- **常見易錯點提醒**：題目規定雨水可以流向高度 **小於或等於** 的相鄰格子，因此逆向爬升時的判斷條件一定要寫 **`>=`（大於等於）**，如果誤寫成嚴格大於 `>`，等高平台上的水流路徑就會被截斷而漏掉正確答案！

---

## **逆向圖論搜尋 / 雙源交集類問題整理**

| 題號 | 題目名稱 | 問題類型 | 解法 |
|---|---|---|---|
| **LeetCode 417** | Pacific Atlantic Water Flow（太平洋大西洋水流問題） | 找出能同時流向 **上/左邊界** 與 **下/右邊界** 的格子 | **雙邊界逆向 DFS / BFS**：分別從兩大洋海岸往 `>=` 高度爬升，取交集 |
| **LeetCode 130** | Surrounded Regions（被圍繞的區域） | 捕獲所有不與四條邊界相連的內部 `'O'` 區域 | **單一整體邊界逆向 DFS / BFS**：從四條邊界標記安全區 `'T'`，其餘 `'O'` 變 `'X'` |
| **LeetCode 1020** | Number of Enclaves（飛地的數量） | 統計無法走出網格邊界的陸地單元格數量 | **邊界逆向 DFS / BFS**：淹沒與四條邊界相連的陸地 `1`，統計剩餘 `1` |
| **LeetCode 286** | Walls and Gates（牆與門） | 求每個空房間到最近一扇門的最短步數 | **多源 BFS**：所有門 `0` 同時入隊向外逐層擴散 |
| **LeetCode 329** | Longest Increasing Path in a Matrix | 求矩陣中嚴格遞增路徑的最長長度 | **記憶化搜尋（DFS + Memoization）**：利用高度單調性構成 DAG（有向無環圖）直接 DP |
| **LeetCode 827** | Making A Large Island（最大人工島） | 最多將一個 `0` 變成 `1`，求能形成的最大島嶼面積 | **連通塊編號 + 面積雜湊表**：先標記各島嶼 ID 與面積，再枚舉每個 `0` 合併四周不同島嶼 |

---

### **進階變形題：如果題目改問「哪一個座標能流向兩大洋，且它的海拔高度最高（或找出所有能流向兩大洋的最大連通分水嶺）」？**

只需在算出 `pac` 與 `atl` 的交集座標後，順便篩選最大高度 `max_h`，或利用 **位元遮罩陣列 `state[r][c] == 3`** 直接過濾出目標座標：

```python
class Solution:
    def highestWatershedCells(self, heights: List[List[int]]) -> List[List[int]]:
        ROWS, COLS = len(heights), len(heights[0])
        # 位元狀態：1 = 可達太平洋, 2 = 可達大西洋, 3 = 兩者皆可達
        state = [[0] * COLS for _ in range(ROWS)]

        def dfs(r: int, c: int, ocean_bit: int) -> None:
            state[r][c] |= ocean_bit
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < ROWS and 0 <= nc < COLS and
                    not (state[nr][nc] & ocean_bit) and
                    heights[nr][nc] >= heights[r][c]
                ):
                    dfs(nr, nc, ocean_bit)

        for c in range(COLS):
            if not (state[0][c] & 1):
                dfs(0, c, 1)
            if not (state[ROWS - 1][c] & 2):
                dfs(ROWS - 1, c, 2)

        for r in range(ROWS):
            if not (state[r][0] & 1):
                dfs(r, 0, 1)
            if not (state[r][COLS - 1] & 2):
                dfs(r, COLS - 1, 2)

        # 找出所有能同時流向兩大洋的格子，並篩選海拔最高者
        both = [(r, c) for r in range(ROWS) for c in range(COLS) if state[r][c] == 3]
        if not both:
            return []
        max_height = max(heights[r][c] for r, c in both)
        return [[r, c] for r, c in both if heights[r][c] == max_height]
```
````