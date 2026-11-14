
LeetCode 130 / LintCode 477 (NeetCode 150)
給你一個 `m x n` 的矩陣 `board` ，由若干字元 `'X'` 和 `'O'` 組成，請你捕獲所有 **被圍繞的區域** ：
- **連接**：一個單元格與水平或垂直方向上相鄰的單元格連接。
- **區域**：連接所有 `'O'` 的單元格形成一個區域。
- **捕獲**：如果一個區域中 **沒有任何單元格** 位於 `board` 矩陣的邊緣，則該區域被視為「被圍繞的」。透過將該區域內的所有 `'O'` 原地替換為 `'X'` 來捕獲被圍繞的區域（無需回傳任何值）。

**樣例 1：**
輸入：
```python
board = [
  ["X", "X", "X", "X"],
  ["X", "O", "O", "X"],
  ["X", "X", "O", "X"],
  ["X", "O", "X", "X"]
]
```
輸出：
```python
[
  ["X", "X", "X", "X"],
  ["X", "X", "X", "X"],
  ["X", "X", "X", "X"],
  ["X", "O", "X", "X"]
]
```
解釋：
中間的三个 `'O'`（座標 `(1,1), (1,2), (2,2)`）四面都被 `'X'` 包圍，因此全部被捕獲並翻轉為 `'X'`。
最下方 `(3,1)` 的 `'O'` 位於矩陣最後一列的邊界上，沒有被完全圍繞，因此保持 `'O'` 不變。

**樣例 2：**
輸入：
```python
board = [["X"]]
```
輸出：
```python
[["X"]]
```
解釋：
網格中沒有任何 `'O'`，無需修改

**樣例 3：**
輸入：
```python
board = [
  ["O", "O"],
  ["O", "O"]
]
```
輸出：
```python
[
  ["O", "O"],
  ["O", "O"]
]
```
解釋：
所有 `'O'` 都位於邊界上，沒有任何區域被 `'X'` 包圍，全部保持 `'O'`

---

### 解法一：邊界逆向廣度優先搜尋（Boundary BFS，避免大型網格遞迴爆棧）

```python
from collections import deque

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        if not board or not board[0]:
            return

        ROWS, COLS = len(board), len(board[0])
        queue = deque()

        # 1. 將四條邊界上的所有 'O' 加入 Queue，並暫時標記為安全區 'T'
        for r in range(ROWS):
            for c in [0, COLS - 1]:
                if board[r][c] == "O":
                    board[r][c] = "T"
                    queue.append((r, c))

        for c in range(COLS):
            for r in [0, ROWS - 1]:
                if board[r][c] == "O":
                    board[r][c] = "T"
                    queue.append((r, c))

        # 2. 從邊界安全區向外擴散，把所有與邊界相連的 'O' 都標記為 'T'
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS and board[nr][nc] == "O":
                    board[nr][nc] = "T"
                    queue.append((nr, nc))

        # 3. 遍歷全圖：剩下的 'O' 必為被圍繞區域（改為 'X'），安全區 'T' 還原為 'O'
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == "O":
                    board[r][c] = "X"
                elif board[r][c] == "T":
                    board[r][c] = "O"
```
pass

### 解法二：邊界逆向深度優先搜尋 + 原地標記（NeetCode 經典最佳解，最精簡）

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        if not board or not board[0]:
            return

        ROWS, COLS = len(board), len(board[0])

        def capture(r: int, c: int) -> None:
            # 越界或不是 'O' 就返回
            if r < 0 or c < 0 or r >= ROWS or c >= COLS or board[r][c] != "O":
                return
            # 將與邊界相連的 'O' 暫時標記為 'T' (Temporary / Safe)
            board[r][c] = "T"
            capture(r + 1, c)
            capture(r - 1, c)
            capture(r, c + 1)
            capture(r, c - 1)

        # 階段一（Reverse Thinking）：只從四條邊界上的 'O' 啟動 DFS 標記安全區
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == "O" and (r in [0, ROWS - 1] or c in [0, COLS - 1]):
                    capture(r, c)

        # 階段二：把所有未受保護的內部 'O' 捕獲為 'X'
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == "O":
                    board[r][c] = "X"

        # 階段三：把之前標記為安全區的 'T' 還原回 'O'
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == "T":
                    board[r][c] = "O"
```

解釋:
step1: 核心逆向思維（Reverse Thinking）：題目要我們找「被 `'X'` 包圍的 `'O'`」，直接判斷每個連通塊是否觸碰邊界很麻煩；反過來想，**「凡是沒被包圍的 `'O'`，一定直接或間接連到矩陣的四條邊界」**。
step2: 假設 `board` 為樣例 1 的 `4 x 4` 矩陣：
  - 第 0 列：`["X", "X", "X", "X"]`
  - 第 1 列：`["X", "O", "O", "X"]`
  - 第 2 列：`["X", "X", "O", "X"]`
  - 第 3 列：`["X", "O", "X", "X"]`
step3: **階段一（邊界掃描與 DFS 標記）**：
  - 檢查四條邊界（`r == 0`, `r == 3`, `c == 0`, `c == 3`），只在 `(3, 1)` 發現一個 `'O'`。
  - 從 `(3, 1)` 呼叫 `capture(3, 1)`：將 `board[3][1]` 改為 `'T'`；檢查四個方向，上方 `(2, 1)` 是 `'X'`、左 `(3, 0)` 是 `'X'`、右 `(3, 2)` 是 `'X'`、下方越界，結束 DFS。
  - 而內部連通塊 `(1, 1), (1, 2), (2, 2)` 因為沒有接觸任何邊界，完全不會被 DFS 觸及，依然維持 `'O'`。
step4: **階段二（捕獲內部 `'O'`）**：
  - 掃描整個矩陣，遇到還是 `'O'` 的格子（即 `(1, 1), (1, 2), (2, 2)`），代表它們無法通往邊界、完全被圍繞，直接將其覆寫為 `'X'`。
step5: **階段三（還原安全區 `'T'`）**：
  - 再次掃描矩陣（階段二與階段三可合併在同一個雙層迴圈中完成），將標記為 `'T'` 的 `(3, 1)` 還原回 `'O'`，完成原地修改。

---

### **LeetCode 130 - Surrounded Regions（被圍繞的區域）**

這道題目是二維網格圖論遍歷中 **「正難則反（邊界逆向搜尋）」** 思想的經典代表題。與一般「遇到島嶼就消滅」的 Flood Fill 題型不同，本題要求區分「封閉區域」與「開放區域」，透過從邊界向內入侵標記安全區，可以大幅簡化程式碼邏輯。

---

## **解法解析**

這道題如果採用「正面突破」的寫法，對每個 `'O'` 連通塊都先做一次搜尋確認是否碰壁、記錄座標列表、若沒碰壁再回頭把座標全部改成 `'X'`，不僅程式碼冗長，還需要額外的 `visited` 與暫存集合。

### **思路**

1. **正難則反（Capture Unsurrounded Regions First）**
   - 矩陣中所有的 `'O'` 只分為兩類：
     - **不安全（會被翻轉為 `'X'`）**：不與任何邊界 `'O'` 相連的內部連通塊。
     - **安全（最終保留為 `'O'`）**：直接位於四條邊界上，或透過相鄰 `'O'` 與邊界連通的區域。
   - 因為所有安全區的源頭都在 **四條邊界** 上，我們只需要以邊界上的 `'O'` 為起點做 DFS / BFS，就能精準找出所有「倖存者」。
2. **三態原地標記法（`'X'`, `'O'`, `'T'`）**
   - 為了不額外開闢 `O(m × n)` 的 `visited` 布林矩陣，我們直接在原矩陣 `board` 上引入第三種暫時狀態字元 **`'T'`**（代表 Temporary / Safe）：
     - `'T'` 既是「該格子受邊界保護」的標記，同時也充當了「該格子已訪問過」的 `visited` 記錄（防止 DFS/BFS 走回頭路陷入無窮迴圈）。
3. **階段二與階段三可合併執行**
   - 當邊界擴散結束後，矩陣中只會剩下 `'X'`、內部未訪問的 `'O'`、以及邊界安全區 `'T'`。此時只需一次雙層迴圈掃描，把 `'O' -> 'X'`、`'T' -> 'O'` 同步處理即可。

---

## **具體舉例**

以樣例 1 的 `4 x 4` 網格為例，觀察三個階段矩陣狀態的演變：

| 階段 | 操作內容 | 矩陣 `board` 狀態變化 |
|---|---|---|
| **初始狀態** | 原始輸入網格 | `X X X X`<br>`X O O X`<br>`X X O X`<br>`X O X X` |
| **階段一（邊界 DFS/BFS）** | 從邊界 `(3, 1)` 的 `'O'` 出發，將相連的 `'O'` 全部標記為安全區 `'T'` | `X X X X`<br>`X O O X`<br>`X X O X`<br>`X T X X` |
| **階段二（捕獲內部區域）** | 全圖掃描，將剩餘未被標記的 `'O'` 全部翻轉為 `'X'` | `X X X X`<br>`X X X X`<br>`X X X X`<br>`X T X X` |
| **階段三（還原安全區）** | 全圖掃描，將暫時標記 `'T'` 還原回 `'O'` | `X X X X`<br>`X X X X`<br>`X X X X`<br>`X O X X` |

---

## **時間與空間複雜度分析**

設矩陣大小為 `M × N`（`M` 列 `N` 行）。

- **時間複雜度：O(M × N)**
  - 階段一掃描四條邊界並執行 DFS / BFS，由於每個 `'O'` 格子一旦被訪問就會立刻改為 `'T'`，因此整個網格中的每個單元格 **最多只會被訪問一次**，耗時 `O(M × N)`。
  - 階段二與階段三掃描全圖替換字元，耗時 `O(M × N)`。
  - 整體時間複雜度為線性 **O(M × N)**。
- **空間複雜度：O(M × N)**
  - 雖然我們使用原地字元 `'T'` 省去了 `visited` 陣列，但演算法仍需要額外空間：
    - **DFS 解法**：最壞情況下（例如整張圖呈蛇形連通的 `'O'`），遞迴呼叫堆疊（Recursion Call Stack）深度可達 `O(M × N)`。
    - **BFS 解法**：最壞情況下（例如網格邊界與內部幾乎全是 `'O'`），Queue 最多同時儲存 `O(M × N)` 個座標。

---

## **其他解法**

### **1. 並查集（Union-Find / Disjoint Set Union）**
- **思路**：
  - 建立一個虛擬節點 `dummy`（編號為 `M * N`），代表「邊界安全區」。
  - 遍歷網格中所有的 `'O'`：
    - 若該 `'O'` 位於四條邊界上，則將其與 `dummy` 合併（`union(r * COLS + c, dummy)`）；
    - 若該 `'O'` 不在邊界上，則將其與上下左右相鄰的 `'O'` 合併。
  - 最後再次遍歷網格，只要某個 `'O'` 與 `dummy` 不在同一個連通分量中（`find(node) != find(dummy)`），就將其翻轉為 `'X'`。
- **時間複雜度**：O(M × N × α(M × N)) ≈ O(M × N)（`α` 為反阿克曼函數，接近常數）
- **空間複雜度**：O(M × N)（並查集的 `parent` 與 `rank` 陣列）

---

## **總結**

- **面試最佳解**：**邊界逆向 DFS / BFS + 原地標記 `'T'`**，時間 `O(M × N)`，程式碼簡潔直觀，面試時最推薦優先寫出。若擔心大型網格在 Python 中觸發最大遞迴深度限制（`RecursionError`），則選用 **解法一（Boundary BFS）**。
- **核心解題模板**：凡是題目涉及二維網格中的 **「封閉區域」、「飛地（Enclaves）」、「無法流出邊界」** 等條件，不要傻傻從內部每個格子做判斷，一律採用 **「從四條邊界向內逆向搜尋」** 的套路，能把複雜問題降維成基礎的連通塊標記。

---

## **邊界逆向搜尋 / 島嶼封閉性類問題整理**

| 題號 | 題目名稱 | 問題類型 | 解法 |
|---|---|---|---|
| **LeetCode 130** | Surrounded Regions（被圍繞的區域） | 將所有不與邊界相連的 `'O'` 區域翻轉為 `'X'` | **邊界逆向 DFS / BFS**：從邊界 `'O'` 標記安全區 `'T'`，其餘 `'O'` 變 `'X'` |
| **LeetCode 1020** | Number of Enclaves（飛地的數量） | 計算有多少個陸地單元格 `1` 無法通過移動離開網格邊界 | **邊界逆向 DFS / BFS**：把與邊界相連的 `1` 全部淹沒成 `0`，最後統計剩餘 `1` 的總數 |
| **LeetCode 1254** | Number of Closed Islands（統計封閉島嶼的數目） | 計算完全被水域 `1` 包圍的陸地 `0` 島嶼（連通塊）有幾個 | **先淹沒邊界連通塊 + 再算島嶼數**：先把邊界相連的 `0` 變 `1`，再用 LeetCode 200 計算剩餘 `0` 連通塊數量 |
| **LeetCode 417** | Pacific Atlantic Water Flow（太平洋大西洋水流問題） | 找出能同時流向太平洋（上/左邊界）與大西洋（下/右邊界）的格子 | **雙邊界逆向 DFS / BFS**：分別從兩大洋邊界往高度大於等於當前格子的方向逆流，取兩個集合交集 |
| **LeetCode 200** | Number of Islands（島嶼數量） | 計算網格中所有 `'1'` 連通塊的總數量（不論是否靠邊界） | **基礎 DFS / BFS Flood Fill**：遇到 `'1'` 計數加一並將整個連通塊淹沒為 `'0'` |
| **LeetCode 695** | Max Area of Island（島嶼的最大面積） | 計算網格中最大連通 `'1'` 區域的格子數量 | **基礎 DFS / BFS**：遍歷連通塊時累計節點數並維護最大值 |

---

### **進階變形題：如果只要求計算「被圍繞的區域總共有幾個格子」（LeetCode 1020 Number of Enclaves）？**

這題與本題邏輯完全一致（只是 `1` 代表陸地、`0` 代表海洋），而且因為不需要還原安全區，連 `'T'` 都不需要——直接把所有與邊界相連的陸地 `1` 全部「淹沒」成 `0`，最後回傳整個矩陣中剩餘 `1` 的總和即可：

```python
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        ROWS, COLS = len(grid), len(grid[0])

        def dfs(r: int, c: int) -> None:
            if r < 0 or c < 0 or r >= ROWS or c >= COLS or grid[r][c] == 0:
                return
            # 直接將與邊界相連的陸地淹沒為海洋 0
            grid[r][c] = 0
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        # 1. 從四條邊界出發，淹沒所有能通往邊界的陸地
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == 1 and (r in [0, ROWS - 1] or c in [0, COLS - 1]):
                    dfs(r, c)

        # 2. 剩餘的所有 1 皆為無法離開邊界的「飛地」，直接加總即可
        return sum(sum(row) for row in grid)
```
````