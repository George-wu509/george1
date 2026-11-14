
LeetCode 36 / LintCode 389
请你判断一个 `9 x 9` 的数独是否有效。只需要**根据以下规则**，验证已经填入的数字是否有效即可。

1. 数字 `1-9` 在每一行只能出现一次。
2. 数字 `1-9` 在每一列只能出现一次。
3. 数字 `1-9` 在每一个以粗实线分隔的 `3 x 3` 宫内只能出现一次。（请参考示例图）

**注意：**
- 一个有效的数独（部分已被填充）不一定是可解的。
- 只需要根据以上规则，验证已经填入的数字是否有效即可。
- 空白格用 `'.'` 表示。

**样例 1：**
输入：
```python
board = [
    ["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]
````

输出：
```Python
True
```

解释： 每一行、每一列、以及 9 个 3x3 九宫格内都没有重复的数字（1-9），因此返回 True。

**样例 2：** 输入：
```Python
board = [
    ["8","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]
```

输出：
```Python
False
```

解释： 除了第一行的第一个数字从 5 改为 8 以外，其余与样例 1 相同。 由于左上角的 3x3 宫内出现了两个 `'8'`（同時第一列也出現了兩個 `'8'`），因此返回 False。

```Python
# 解法一：一次遍歷 + Hash Set（最直觀推薦）
import collections

class Solution:
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        rows = collections.defaultdict(set)
        cols = collections.defaultdict(set)
        boxes = collections.defaultdict(set)  # key 為 (r // 3, c // 3)

        for r in range(9):
            for c in range(9):
                val = board[r][c]
                if val == ".":
                    continue
                
                box_idx = (r // 3, c // 3)
                if (val in rows[r] or 
                    val in cols[c] or 
                    val in boxes[box_idx]):
                    return False
                
                rows[r].add(val)
                cols[c].add(val)
                boxes[box_idx].add(val)
                
        return True
```


```Python
# 解法二：位元運算 Bitmask（極致空間優化）
class Solution:
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        rows = [0] * 9
        cols = [0] * 9
        boxes = [0] * 9

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                
                # 將字元 '1'~'9' 轉為 0~8 的位移量
                num = int(board[r][c]) - 1
                mask = 1 << num
                box_idx = (r // 3) * 3 + (c // 3)
                
                if (rows[r] & mask) or (cols[c] & mask) or (boxes[box_idx] & mask):
                    return False
                
                rows[r] |= mask
                cols[c] |= mask
                boxes[box_idx] |= mask
                
        return True
```

解釋: step1: 建立三個雜湊表（`rows`, `cols`, `boxes`），分別記錄每一行、每一列、每個 3x3 九宮格出現過的數字。 step2: 使用雙層迴圈遍歷整個 `9 x 9` 棋盤（`r` 從 0 到 8，`c` 從 0 到 8）。 step3: 遇到空白格 `'.'` 時直接 `continue` 跳過，只檢查已填入數字的格子。 step4: 計算當前格子所屬的 3x3 九宮格座標 `box_idx = (r // 3, c // 3)`（例如 `(4, 5)` 整除 3 後落在 `(1, 1)` 正中央的九宮格）。 step5: 檢查該數字 `val` 是否已經存在於 `rows[r]`、`cols[c]` 或 `boxes[box_idx]` 中；若任一為真，代表重複，立即回傳 `False`。 step6: 若無重複，將 `val` 分別加入這三個對應的 `set` 中，全部掃描完畢皆無衝突則回傳 `True`。

解釋: 這題的核心在於「如何將二維座標 `(r, c)` 映射到對應的 `3 x 3` 九宮格編號」。

透過整數除法 `(r // 3, c // 3)`，可以把 `0~8` 的行列索引壓縮成 `0~2` 的區塊座標，剛好對應 `3 x 3 = 9` 個獨立的九宮格。

### **LeetCode 36 - Valid Sudoku（有效的數獨）**

這道題目是典型的 **雜湊表（Hash Table）與矩陣索引映射** 問題。題目只要求驗證「目前已填入的數字」是否違反數獨規則，**不需要判斷該數獨最終是否有解**，因此不需要用到回溯法（Backtracking）。

## **解法解析**

這道題可以透過 **一次遍歷（One-Pass）搭配 Hash Set** 來求解。

### **思路**

1. **拆解三個驗證維度**
    
    - 對於棋盤上的每一個數字 `board[r][c]`，它同時受到三個規則限制：
        
        - **所屬行**：第 `r` 行（共 9 行，`0 ~ 8`）
            
        - **所屬列**：第 `c` 列（共 9 列，`0 ~ 8`）
            
        - **所屬 3x3 區塊**：將 `9 x 9` 切分為 `3 x 3` 個大格子，座標為 `(r // 3, c // 3)`
            
2. **九宮格座標映射公式**
    
    - 二維元組表示法：`(r // 3, c // 3)`，範圍從 `(0, 0)` 到 `(2, 2)`，最直觀不易寫錯。
        
    - 一維陣列索引法：`(r // 3) * 3 + (c // 3)`，範圍從 `0` 到 `8`，適合搭配固定長度陣列或 Bitmask 使用。
        
3. **代碼執行流程**
    
    - 初始化 `rows`、`cols`、`boxes` 三個 `defaultdict(set)`。
        
    - 逐格掃描 `board[r][c]`，遇到 `'.'` 略過。
        
    - 查詢 `val` 是否已在 `rows[r]`、`cols[c]` 或 `boxes[(r // 3, c // 3)]` 中出現過：
        
        - 若有 → 違反規則，直接回傳 `False`。
            
        - 若無 → 將 `val` 記錄進這三個集合中。
            
    - 遍歷結束未發現任何重複，回傳 `True`。
        

## **具體舉例**

以 **样例 2** 的前幾步為例，追蹤集合內的狀態變化：

|**座標 (r, c)**|**數值 val**|**九宮格 box_idx**|**rows[r] 狀態**|**cols[c] 狀態**|**boxes[box_idx] 狀態**|**結果**|
|---|---|---|---|---|---|---|
|`(0, 0)`|`'8'`|`(0, 0)`|`{'8'}`|`{'8'}`|`{'8'}`|通過|
|`(0, 1)`|`'3'`|`(0, 0)`|`{'8', '3'}`|`{'3'}`|`{'8', '3'}`|通過|
|`(0, 4)`|`'7'`|`(0, 1)`|`{'8', '3', '7'}`|`{'7'}`|`{'7'}`|通過|
|`(1, 0)`|`'6'`|`(0, 0)`|`{'6'}`|`{'8', '6'}`|`{'8', '3', '6'}`|通過|
|`(2, 1)`|`'9'`|`(0, 0)`|`{'9'}`|`{'3', '9'}`|`{'8', '3', '6', '9'}`|通過|
|`(2, 2)`|`'8'`|`(0, 0)`|`{'9', '8'}`|`{'8'}`|`'8'` 已存在！|**回傳 `False`**|

## **時間與空間複雜度分析**

- **時間複雜度：O(1)**（若推廣至 `N x N` 棋盤則為 **O(N²)**）
    
    - 數獨棋盤大小固定為 `9 x 9 = 81` 格，每格最多執行一次查詢與三次插入（皆為 `O(1)`），總操作次數有常數上限，因此時間複雜度為 **O(1)**。
        
- **空間複雜度：O(1)**（若推廣至 `N x N` 棋盤則為 **O(N²)**）
    
    - `rows`、`cols`、`boxes` 各自最多儲存 `9 x 9 = 81` 個元素，使用空間為固定常數上限，因此空間複雜度為 **O(1)**。
        

## **其他解法**

### **1. 固定長度二維陣列（Fixed Array）**

- **思路**：
    
    - 不使用 `set`，改用 `rows = [[False] * 9 for _ in range(9)]` 記錄每個數字（`0~8`）是否出現過。
        
- **時間複雜度**：O(1)
    
- **空間複雜度**：O(1)（避免 Hash Set 的雜湊開銷，實際執行速度更快）
    

### **2. 位元運算（Bitmasking）**

- **思路**：
    
    - 用一個 9-bit 的整數（`0` 到 `511`）代替整個 `set` 或布林陣列。數字 `d` 出現時，將第 `d-1` 個 bit 設為 `1`（`mask = 1 << (d - 1)`），透過 `&` 檢查重複、`|` 記錄狀態。
        
- **時間複雜度**：O(1)
    
- **空間複雜度**：O(1)（僅需 27 個整數，空間開銷最低）
    

## **總結**

- **面試最推薦解法**：**一次遍歷 + `defaultdict(set)`**，邏輯最清晰、可讀性最高，且不容易寫出索引越界 Bug。
    
- **極致效能優化解法**：**Bitmask 位元運算**，僅用三個長度為 9 的整數陣列即可完成狀態壓縮，適合需要處理大規模重複驗證（如解數獨）的場景。
    
- **關鍵記憶點**：3x3 九宮格的索引轉換公式 **`(r // 3, c // 3)`**。
    

## **LeetCode 數獨與矩陣驗證類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 36**|Valid Sudoku（有效的數獨）|**2D 矩陣規則驗證（行、列、3x3 區塊）**|**Hash Set / Bitmask 一次遍歷**，映射 `(r // 3, c // 3)`|
|**LeetCode 37**|Sudoku Solver（解數獨）|**2D 矩陣約束滿足與搜尋填格**|**回溯法（Backtracking）+ Bitmask / Set 剪枝**|
|**LeetCode 73**|Set Matrix Zeroes（矩陣置零）|**2D 矩陣狀態標記與原地修改**|**首行首列作為標記位（O(1) 空間優化）**|
|**LeetCode 48**|Rotate Image（旋轉圖像）|**2D 矩陣座標變換與原地旋轉**|**先轉置矩陣（Transpose）再左右翻轉（Reverse）**|
|**LeetCode 54**|Spiral Matrix（螺旋矩陣）|**2D 矩陣邊界模擬遍歷**|**維護上下左右四個邊界指標收縮遍歷**|
|**LeetCode 2133**|Check if Every Row and Column Contains All Numbers|**N x N 矩陣行列完整性驗證**|**每行每列建立 Set 檢查長度是否等於 N**|

### **進階變形：若題目改為 `N x N` 的廣義數獨（例如 `16 x 16`，區塊大小為 `√N x √N`）該如何寫？**

Python

```
import math
import collections

class Solution:
    def isValidGeneralSudoku(self, board: list[list[str]]) -> bool:
        n = len(board)
        box_size = int(math.isqrt(n)) # 例如 9 -> 3，16 -> 4
        
        rows = collections.defaultdict(set)
        cols = collections.defaultdict(set)
        boxes = collections.defaultdict(set)

        for r in range(n):
            for c in range(n):
                val = board[r][c]
                if val == ".":
                    continue
                
                # 動態依據區塊邊長 box_size 計算九宮格座標
                box_idx = (r // box_size, c // box_size)
                
                if val in rows[r] or val in cols[c] or val in boxes[box_idx]:
                    return False
                
                rows[r].add(val)
                cols[c].add(val)
                boxes[box_idx].add(val)
                
        return True
```

```

<ElicitationsGroup message="繼續整理下一題 Blind 75 / NeetCode 150 筆記：">
  <Elicitation label="LeetCode 128. 最長連續序列" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 128. Longest Consecutive Sequence（最長連續序列）的完整筆記。"/>
  <Elicitation label="LeetCode 238. 除自身以外數組的乘積" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 238. Product of Array Except Self（除自身以外數組的乘積）的完整筆記。"/>
  <Elicitation label="LeetCode 37. 解數獨（進階延伸）" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 37. Sudoku Solver（解數獨）的完整筆記。"/>
</ElicitationsGroup>
```