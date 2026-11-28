
|                        |     |
| ---------------------- | --- |
| [[#### Word Search解法]] |     |
| [[#### 為何不推薦使用BFS?]]   |     |
|                        |     |
#### Word Search解法
```

```
LeetCode 79 / LintCode 123 (Blind 75 + NeetCode 150)
給定一個 `m x n` 的二維字元網格 `board` 和一個字串單字 `word` 。如果 `word` 存在於網格中，返回 `true` ；否則，返回 `false` 。
單字必須按照字母順序，通過相鄰的單元格內的字母構成，其中「相鄰」單元格是那些水平相鄰或垂直相鄰的單元格。同一個單元格內的字母不允許被重複使用。

**樣例 1：**
輸入：
```python
board = [
  ["A", "B", "C", "E"],
  ["S", "F", "C", "S"],
  ["A", "D", "E", "E"]
], word = "ABCCED"
```
輸出：
```python
True
```
解釋：
從 `(0,0)` 的 `'A'` 出發，依序向右 `(0,1) 'B'` -> `(0,2) 'C'` -> 向下 `(1,2) 'C'` -> `(2,2) 'E'` -> 向左 `(2,1) 'D'`，成功拼出 `"ABCCED"`

**樣例 2：**
輸入：
```python
board = [
  ["A", "B", "C", "E"],
  ["S", "F", "C", "S"],
  ["A", "D", "E", "E"]
], word = "SEE"
```
輸出：
```python
True
```
解釋：
從 `(1,3)` 的 `'S'` 出發 -> 向下 `(2,3) 'E'` -> 向左 `(2,2) 'E'`，成功拼出 `"SEE"`

**樣例 3：**
輸入：
```python
board = [
  ["A", "B", "C", "E"],
  ["S", "F", "C", "S"],
  ["A", "D", "E", "E"]
], word = "ABCB"
```
輸出：
```python
False
```
解釋：
走到 `(0,2)` 的 `'C'` 後，雖然左邊 `(0,1)` 是 `'B'`，但該格子已經在路徑中被使用過，不可重複走訪，因此無解

---

### 解法一：標準 DFS + `visited` 集合回溯法（NeetCode 經典寫法，不修改原陣列）
```python
class Solution:
    def exist(self, board, word):
        rows, cols = len(board), len(board[0])
        path = set()(r, c)
        
        for r in range(rows):
            for c in range(cols):
                if self.dfs(board, word, r, c, 0):
                    return True
        return False

    def dfs(self, board, word, r, c, i):
        if i == len(word):
            return True
        if (
            r < 0 or c < 0 or
            r >= rows or c >= cols or
            board[r][c] != word[i] or
            (r, c) in path
        ):
            return False

        path.add((r, c))
        res = (
            dfs(r + 1, c, i + 1) or
            dfs(r - 1, c, i + 1) or
            dfs(r, c + 1, i + 1) or
            dfs(r, c - 1, i + 1)
        )
        path.remove((r, c))
        return res
```
解說version
```python
class Solution:
    def exist(self, board, word):
        rows, cols = len(board), len(board[0])
        path = set()(r, c) # 記錄當前DFS路徑已走過的座標 
        
        for r in range(rows):
            for c in range(cols):
                if self.dfs(board, word, r, c, 0):
                    return True
        return False

    def dfs(self, board, word, r, c, i):
	    # 1. 成功配對完 word 的所有字元
        if i == len(word):
            return True
        # 2. 越界、字元不符，或該格子已在當前路徑中使用過
        if (
            r < 0 or c < 0 or
            r >= rows or c >= cols or
            board[r][c] != word[i] or
            (r, c) in path
        ):
            return False

		# 做選擇：將當前座標加入路徑集合
        path.add((r, c))
        # 向四個方向探索下一個字元 word[i + 1]
        res = (
            dfs(r + 1, c, i + 1) or
            dfs(r - 1, c, i + 1) or
            dfs(r, c + 1, i + 1) or
            dfs(r, c - 1, i + 1)
        )
        #回溯：離開這條分支前移除座標，讓其他分支能正常使用
        path.remove((r, c))
        return res
```


### 解法二：原地標記回溯 + 頻率與首尾反轉剪枝（面試最佳解，實務執行最快）

```python
from collections import Counter

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        ROWS, COLS = len(board), len(board[0])
        
        # 剪枝 1：若單字長度大於網格總格數，不可能存在
        if len(word) > ROWS * COLS:
            return False

        # 剪枝 2：統計網格與單字的字元頻率，若某字元數量不足直接回傳 False
        board_cnt = Counter(ch for row in board for ch in row)
        word_cnt = Counter(word)
        for ch, cnt in word_cnt.items():
            if board_cnt[ch] < cnt:
                return False

        # 剪枝 3（關鍵加速）：若單字「結尾字元」在網格中出現次數比「開頭字元」少，將 word 反轉從稀疏端開始搜！
        if board_cnt[word[0]] > board_cnt[word[-1]]:
            word = word[::-1]

        def dfs(r: int, c: int, i: int) -> bool:
            if i == len(word):
                return True
            if r < 0 or c < 0 or r >= ROWS or c >= COLS or board[r][c] != word[i]:
                return False

            # 原地標記（In-place Marking）：暫存原字元並改為 '#'，省去 set 的雜湊開銷
            temp = board[r][c]
            board[r][c] = "#"

            found = (
                dfs(r + 1, c, i + 1) or
                dfs(r - 1, c, i + 1) or
                dfs(r, c + 1, i + 1) or
                dfs(r, c - 1, i + 1)
            )

            # 回溯還原現場
            board[r][c] = temp
            return found

        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == word[0] and dfs(r, c, 0):
                    return True
        return False
```

解釋:
step1: 先進行前置剪枝檢查：統計 `board` 內每個字母的總數，確認足以組成 `word`；接著比較 `word[0]` 與 `word[-1]` 在網格中的出現次數（例如搜尋 `"AAAAAB"` 時，`'A'` 到處都是但 `'B'` 只有 1 個，反轉成 `"BAAAAA"` 從 `'B'` 開始搜可省下大量無效分支）。
step2: 假設 `board = [["A","B","C","E"], ["S","F","C","S"], ["A","D","E","E"]]`，`word = "SEE"`（`'S'` 出現 2 次，`'E'` 出現 3 次，不反轉）。
step3: 雙層迴圈掃描網格尋找開頭字元 `'S'`：
  - 第一個 `'S'` 在 `(1, 0)`：呼叫 `dfs(1, 0, 0)`，將 `board[1][0]` 暫改為 `'#'`，檢查四個方向尋找 `word[1] = 'E'`：
    - 上 `(0, 0) 'A'`、下 `(2, 0) 'A'`、右 `(1, 1) 'F'`、左越界，皆不符合 `'E'`。
    - 觸發回溯，將 `board[1][0]` 還原為 `'S'`，回傳 `False`。
step4: 繼續掃描到第二個 `'S'` 在 `(1, 3)`：呼叫 `dfs(1, 3, 0)`：
  - 配對 `word[0] = 'S'` 成功，將 `board[1][3]` 改為 `'#'`，搜尋 `word[1] = 'E'`。
  - 上 `(0, 3)` 是 `'E'`：進入 `dfs(0, 3, 1)`，將 `board[0][3]` 改為 `'#'`，四周找 `word[2] = 'E'` 失敗，回溯還原 `board[0][3] = 'E'`。
  - 下 `(2, 3)` 是 `'E'`：進入 `dfs(2, 3, 1)`，將 `board[2][3]` 改為 `'#'`，搜尋 `word[2] = 'E'`。
step5: 在 `(2, 3)` 往左探索 `(2, 2)`：
  - 進入 `dfs(2, 2, 2)`，發現 `board[2][2] == 'E' == word[2]`，將 `board[2][2]` 改為 `'#'`，繼續呼叫 `dfs(..., i = 3)`。
  - 進入下一層發現 `i == len(word) == 3`，代表所有字元皆已成功串連，立刻回傳 `True`！
step6: 一路向上回傳 `True`（同時還原沿途的 `'#'`），外層迴圈接收到 `True` 後直接提前結束並回傳 `True`。

---

### **LeetCode 79 - Word Search（單字搜尋）**

這道題目是二維矩陣上結合 **深度優先搜尋（DFS）** 與 **回溯演算法（Backtracking）** 的標準模板題。與一般「島嶼數量」類型的圖論遍歷不同，本題要求尋找一條**有順序的連續路徑**，因此走訪狀態必須隨著遞迴退回而還原。

---

## **解法解析**

這道題最容易出錯的地方在於**「已訪問狀態的還原時機」**，許多初學者會誤把本題寫成普通的 Flood Fill（只標記 `visited` 卻沒有在回溯時清除），導致失敗的分支把格子「永久鎖死」，讓其他正確路徑無法經過。

### **思路**

1. **為什麼必須做「回溯還原（Backtracking）」？**
   - 假設我們要找 `"ABCESE"`，當走到 `'C'` 時，可能先錯誤地往右走到 `'E'` 發現後面是死路；退回 `'C'` 後，改往下走到 `'C' -> 'E' -> 'S' -> 'E'` 才走通。
   - 如果剛剛錯誤試探的那格 `'E'` 沒有在退出遞迴時把 `visited`（或 `'#'`）還原回來，正確路徑經過這格時就會被誤判為「已使用」而失敗！
   - 因此，**「進入節點時標記、離開節點（四個方向都試完）時還原」** 是本題的核心紀律。
2. **原地標記法（In-Place Modification）取代 `visited` 集合**
   - 題目限定網格中只會出現大小寫英文字母，我們可以在走訪 `(r, c)` 時先用區域變數 `temp = board[r][c]` 備份原字元，再把 `board[r][c]` 暫時改寫為特殊符號 `'#'`（一定不會等於任何 `word[i]`），離開該層遞迴前再執行 `board[r][c] = temp` 恢復原狀。
   - 這樣做完全不需要建立 `O(m × n)` 的 `visited` 陣列或 Python `set()`，既節省記憶體，又省去 tuple 雜湊計算的時間開銷。
3. **極致效能優化：首尾字元頻率反轉剪枝**
   - 搜尋樹的分支數量在**前幾層**影響最大。如果測試資料故意設計成 `board` 裡有幾百個 `'A'`，而 `word = "AAAAAAAAB"`：
     - 從 `'A'` 開始搜會引發數百次深層無效遞迴；
     - 若事先統計字元數量，發現結尾 `'B'` 比開頭 `'A'` 稀少，直接執行 `word = word[::-1]` 從 `'B'` 開始倒著搜，能瞬間砍掉 99% 的無效分支！

---

## **具體舉例**

以樣例 1 `word = "ABCCED"` 為例，觀察從 `(0, 0)` 出發的 DFS 與回溯還原軌跡：

| 步驟 | 當前座標 `(r, c)` | 匹配字元 `word[i]` | 網格原地標記狀態 | 動作與決策說明 |
|---|---|---|---|---|
| 1 | `(0, 0)` | `i = 0 ('A')` | `board[0][0] = '#'` | 匹配 `'A'`，嘗試下 `(1,0) 'S'` 失敗，轉向右 `(0,1)` |
| 2 | `(0, 1)` | `i = 1 ('B')` | `board[0][1] = '#'` | 匹配 `'B'`，嘗試下 `(1,1) 'F'` 失敗，轉向右 `(0,2)` |
| 3 | `(0, 2)` | `i = 2 ('C')` | `board[0][2] = '#'` | 匹配 `'C'`，先試右 `(0,3) 'E'` 不符 `word[3]='C'`，轉向下 `(1,2)` |
| 4 | `(1, 2)` | `i = 3 ('C')` | `board[1][2] = '#'` | 匹配 `'C'`，試左 `(1,1) 'F'` 失敗、試右 `(1,3) 'S'` 失敗，轉向下 `(2,2)` |
| 5 | `(2, 2)` | `i = 4 ('E')` | `board[2][2] = '#'` | 匹配 `'E'`，試右 `(2,3) 'E'` 不符 `word[5]='D'`，轉向左 `(2,1)` |
| 6 | `(2, 1)` | `i = 5 ('D')` | `board[2][1] = '#'` | 匹配 `'D'`，下一步進入 `i = 6 == len(word)`，**觸發終止條件回傳 `True`** |

---

## **時間與空間複雜度分析**

設網格大小為 `M × N`（`M` 列 `N` 行），單字 `word` 的長度為 `L`。

- **時間複雜度：O(M × N × 3^L)**
  - 外層迴圈最壞情況下需要以網格中的每一個格子作為起點，共 `M × N` 個起點。
  - 每次啟動 DFS 時，第一步最多有 `4` 個方向可選；但從第二步開始，因為不能走回剛剛進來的那個格子（已標記為 `'#'`），每個節點最多只剩 **`3` 個有效分支**，遞迴最大深度為單字長度 `L`。
  - 因此單次 DFS 最壞時間為 `O(3^L)`，整體時間複雜度為 **O(M × N × 3^L)**（加上頻率與反轉剪枝後，實際運行速度極快）。
- **空間複雜度：O(L)**
  - 使用原地修改 `board[r][c] = '#'` 代替額外的 `visited` 資料結構，唯一額外空間來自遞迴呼叫堆疊（Recursion Call Stack）。
  - 遞迴深度最大不會超過單字長度 `L`（且 `L <= M × N`），因此空間複雜度為 **O(L)**。

---

## **其他解法**

### **1. 使用獨立 `visited` 二維布林陣列（適合多執行緒或禁止修改輸入陣列時）**
- **思路**：
  - 若面試官要求「不能修改傳入的 `board` 陣列（Read-Only Input）」，則建立 `visited = [[False] * COLS for _ in range(ROWS)]`，在進入 DFS 時設 `visited[r][c] = True`，回溯時設回 `False`。相比 `set()` 儲存座標 tuple，二維布林陣列存取為純索引操作，常數時間更小。
- **時間複雜度**：O(M × N × 3^L)
- **空間複雜度**：O(M × N)（`visited` 二維陣列佔用 `M × N` 空間 + `O(L)` 遞迴堆疊）

---

## **總結**

- **面試最佳解**：**原地標記回溯 + 頻率與首尾反轉剪枝**，時間 `O(M × N × 3^L)`，空間 `O(L)`。
- **解題兩大核心重點**：
  1. **狀態還原**：這不是一次性的島嶼塗色，失敗路徑走過的格子必須在回溯時恢復原字元，否則會干擾其他分支。
  2. **剪枝意識**：在指數級搜尋題中，加入「長度檢查、字元數量檢查、首尾出現頻率比較與反轉」這三行前置剪枝，能讓執行時間從勉強通過直接躍升至擊敗 99% 提交。

---

## **二維網格 DFS / 字串搜尋類問題整理**

| 題號 | 題目名稱 | 問題類型 | 解法 |
|---|---|---|---|
| **LeetCode 79** | Word Search（單字搜尋） | 在二維字元網格中搜尋 **單一單字** 是否存在 | **DFS + 回溯還原狀態**（原地標記 `'#'` + 首尾頻率反轉剪枝） |
| **LeetCode 212** | Word Search II（單字搜尋 II） | 在二維字元網格中同時搜尋 **一組單字列表 `words`** | **前綴樹（Trie）+ 網格 DFS 回溯**（將所有單字建入 Trie 同步搜尋，找到即剪枝刪除葉節點） |
| **LeetCode 200** | Number of Islands（島嶼數量） | 計算網格中連通 `'1'` 的區塊數量 | **基礎 DFS / BFS Flood Fill**（走過直接將 `'1'` 改為 `'0'`，**不需要**回溯還原） |
| **LeetCode 130** | Surrounded Regions（被圍繞的區域） | 找出不與邊界相連的 `'O'` 區域並翻轉為 `'X'` | **邊界逆向 DFS**：從四條邊界的 `'O'` 出發標記為 `'T'`，最後統一將剩餘 `'O'` 變 `'X'`、`'T'` 還原 `'O'` |
| **LeetCode 329** | Longest Increasing Path in a Matrix | 求網格中嚴格遞增路徑的最大長度 | **記憶化搜尋（DFS + Memoization）**：因嚴格遞增保證無環，無需 `visited` 也不需回溯 |
| **LeetCode 980** | Unique Paths III（不同路徑 III） | 從起點走到終點，且必須 **恰好走過每一個無障礙格子一次** | **DFS + 回溯 + 剩餘空格計數器**（與 LeetCode 79 同樣需要原地標記並回溯還原） |

---

### **進階變形題：如果要在同一個網格中同時搜尋「上千個單字」（LeetCode 212 Word Search II）？**

如果對每個單字都呼叫一次 LeetCode 79 的 `exist()`，會重複掃描網格數千次導致超時。最佳做法是先把所有目標單字存入 **前綴樹（Trie）**，在網格上只跑一次 DFS 回溯，順著 Trie 節點往下走，一次就能同時比對所有前綴相同的單字：

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None  # 若為單字結尾，直接儲存完整單字

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # 1. 將所有目標單字建立成 Trie
        root = TrieNode()
        for w in words:
            node = root
            for ch in w:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.word = w

        ROWS, COLS = len(board), len(board[0])
        res = []

        def dfs(r: int, c: int, parent: TrieNode) -> None:
            ch = board[r][c]
            curr_node = parent.children[ch]

            # 找到完整單字，加入答案並將 word 設為 None 避免重複收錄
            if curr_node.word:
                res.append(curr_node.word)
                curr_node.word = None

            # 原地標記已走訪
            board[r][c] = "#"
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS and board[nr][nc] in curr_node.children:
                    dfs(nr, nc, curr_node)
            # 回溯還原現場
            board[r][c] = ch

            # 葉節點剪枝優化：若該 Trie 分支已無剩餘單字，直接刪除避免重複走訪
            if not curr_node.children:
                parent.children.pop(ch)

        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] in root.children:
                    dfs(r, c, root)

        return res
```




#### 為何不推薦使用BFS?
```
在leetcode word search question(給定一個 `m x n` 的二維字元網格 `board` 和一個字串單字 `word` 。如果 `word` 存在於網格中，返回 `true` ；否則，返回 `false`)以coding interview來說是否也推薦使用BFS method? 或者有甚麼理由建議使用DFS? 
```
在 Coding Interview 中，針對 LeetCode 79 (Word Search)，**強烈不推薦使用 BFS，標準且最佳的解法是 DFS（搭配 Backtracking 回溯法）**。

### 為什麼面試推薦 DFS 而非 BFS？

這題的核心限制是：**「同一條搜尋路徑中，每個格子只能使用一次，但不同分支可以重複使用同一格。」** 這個限制讓 DFS 與 BFS 在效能與實作上產生巨大落差：

|**比較維度**|**DFS + Backtracking（推薦）**|**BFS（不推薦）**|
|---|---|---|
|**狀態與空間複雜度**|只需維護一條當前路徑。可直接原地修改 `board`（如改成 `'#'`）並在回溯時還原，額外空間僅需遞迴深度 **$O(L)$**。|Queue 中同時存在多條互不干擾的分支，**每個節點都必須攜帶一份獨立的 `visited` 集合**，複製與儲存成本極高（最壞達 $O(M \cdot N \cdot 3^L \cdot L)$）。|
|**搜尋效率（提早終止）**|沿單一方向一路探底，只要有一條路走通就能立刻 `return True`，大幅節省時間。|必須像水波一樣把長度為 $1, 2, \dots, L-1$ 的**所有**可能前綴全部展開，才能觸及長度為 $L$ 的答案，在測資較大時極易 TLE 或 MLE。|
|**面試擴充性**|程式碼簡潔直覺，且能無縫銜接進階題 **Word Search II (LeetCode 212, DFS + Trie)**。|難以結合 Trie 進行大規模多單字剪枝。|

> **面試建議：** 除非面試官刻意追問「如果不用遞迴/DFS，改用 BFS 該怎麼處理狀態隔離？」，否則一律直接寫 DFS。

### 如果要用 BFS：Python 實作

使用 BFS 時，不能建立全域共用的 `visited`（否則 A 分支走過的格子會擋住 B 分支），必須把 `(row, col, index, visited_set)` 一起打包放進 `deque` 中：

```Python
from collections import deque
from typing import List


class Solution:

  def exist(self, board: List[List[str]], word: str) -> bool:
    if not board or not board[0]:
      return False

    rows, cols = len(board), len(board[0])
    queue = deque()

    # 1. 找出所有符合 word[0] 的起點，各自建立獨立的 visited set
    for r in range(rows):
      for c in range(cols):
        if board[r][c] == word[0]:
          if len(word) == 1:
            return True
          # 狀態格式: (目前列, 目前行, 目前匹配到的 word 索引, 此路徑已走過的座標集合)
          queue.append((r, c, 0, {(r, c)}))

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # 2. 層序擴展搜尋
    while queue:
      r, c, idx, visited = queue.popleft()

      for dr, dc in directions:
        nr, nc = r + dr, c + dc

        # 檢查邊界、是否已在此路徑走過、以及字元是否符合下一個字母
        if (
            0 <= nr < rows
            and 0 <= nc < cols
            and (nr, nc) not in visited
            and board[nr][nc] == word[idx + 1]
        ):
          # 若已經匹配到 word 的最後一個字元，直接回傳 True
          if idx + 1 == len(word) - 1:
            return True

          # 必須複製一份新的 set 給新分支，避免污染其他分支
          new_visited = visited | {(nr, nc)}
          queue.append((nr, nc, idx + 1, new_visited))

    return False
```

### 範例逐步拆解（Step-by-Step Trace）

給定輸入：

- `board`:
    
    - Row 0: `(0,0)='A'`, `(0,1)='B'`, `(0,2)='C'`, `(0,3)='E'`
        
    - Row 1: `(1,0)='S'`, `(1,1)='F'`, `(1,2)='C'`, `(1,3)='S'`
        
    - Row 2: `(2,0)='A'`, `(2,1)='D'`, `(2,2)='E'`, `(2,3)='E'`
        
- `word = "ABCCED"`（索引 `0:'A', 1:'B', 2:'C', 3:'C', 4:'E', 5:'D'`）
    

#### 步驟 1：初始化 Queue（尋找 `'A'`，`idx = 0`）

掃描整個網格，找到兩個 `'A'`，分別建立獨立的起點狀態放入 Queue：

- `queue` 初始狀態：
    
    1. `(0, 0, idx=0, visited={(0,0)})`
        
    2. `(2, 0, idx=0, visited={(2,0)})`
        

#### 步驟 2：第一層展開（尋找 `'B'`，`idx = 1`）

- **Pop `(0, 0, idx=0, {(0,0)})`**：
    
    - 檢查上下左右：`下(1,0)='S'`（不符）、`右(0,1)='B'`（**符合**）。
        
    - **Push** `(0, 1, idx=1, visited={(0,0), (0,1)})`。
        
- **Pop `(2, 0, idx=0, {(2,0)})`**：
    
    - 檢查上下左右：`上(1,0)='S'`（不符）、`右(2,1)='D'`（不符）。
        
    - 此分支無路可走，自然淘汰。
        

#### 步驟 3：第二層展開（尋找 `'C'`，`idx = 2`）

- **Pop `(0, 1, idx=1, {(0,0), (0,1)})`**：
    
    - 檢查上下左右：
        
        - `左(0,0)`：已在 `visited` 中，跳過。
            
        - `下(1,1)='F'`：不符。
            
        - `右(0,2)='C'`：**符合**。
            
    - **Push** `(0, 2, idx=2, visited={(0,0), (0,1), (0,2)})`。
        

#### 步驟 4：第三層展開（尋找 `'C'`，`idx = 3`）

- **Pop `(0, 2, idx=2, {(0,0), (0,1), (0,2)})`**：
    
    - 檢查上下左右：
        
        - `左(0,1)`：已在 `visited` 中，跳過。
            
        - `右(0,3)='E'`：不符。
            
        - `下(1,2)='C'`：**符合**。
            
    - **Push** `(1, 2, idx=3, visited={(0,0), (0,1), (0,2), (1,2)})`。
        

#### 步驟 5：第四層展開（尋找 `'E'`，`idx = 4`）

- **Pop `(1, 2, idx=3, {(0,0), (0,1), (0,2), (1,2)})`**：
    
    - 檢查上下左右：
        
        - `上(0,2)`：已在 `visited` 中，跳過。
            
        - `左(1,1)='F'`、`右(1,3)='S'`：皆不符。
            
        - `下(2,2)='E'`：**符合**。
            
    - **Push** `(2, 2, idx=4, visited={(0,0), (0,1), (0,2), (1,2), (2,2)})`。
        

#### 步驟 6：第五層展開（尋找 `'D'`，`idx = 5`）

- **Pop `(2, 2, idx=4, {(0,0), (0,1), (0,2), (1,2), (2,2)})`**：
    
    - 檢查上下左右：
        
        - `上(1,2)`：已在 `visited` 中，跳過。
            
        - `右(2,3)='E'`：不符。
            
        - `左(2,1)='D'`：**符合**！
            
    - 此時 `idx + 1 == 5`，等於 `len(word) - 1`，代表整個單字 `"ABCCED"` 匹配完成，直接回傳 **`True`**。