
LeetCode 678 / LintCode 1089 給定一個只包含三種字元：`（`、`）` 和 `*` 的字串 `s`，請寫一個函式來判斷這個字串是否為**有效的括號字串**。

有效括號字串的規則如下：

1. 任何左括號 `(` 必須有對應的右括號 `)`。
2. 任何右括號 `)` 必須有對應的左括號 `(`。
3. 左括號 `(` 必須在對應的右括號 `)` 之前。
4. `*` 可以被視為單個右括號 `)`、單個左括號 `(`，或者一個空字串 `""`。
5. 空字串也被視為有效。

**样例 1：** 输入：
```Python
s = "()"
```

输出：
```Python
True
```

解释： 標準的一對左右括號，直接匹配成功。

**样例 2：** 输入：
```Python
s = "(*)"
```

输出：
```Python
True
```

解释： 中間的 `*` 可以當作空字串 `""`，字串變成 `"()"`，為有效括號字串。

**样例 3：** 输入：
```Python
s = "(*))"
```

输出：
```Python
True
```

解释： 將 `*` 視為左括號 `(`，字串變成 `"(())"`，兩對括號完美匹配。


```Python
class Solution:
    def checkValidString(self, s: str) -> bool:
        left_min = 0  # 未匹配左括號 '(' 的最少可能數量
        left_max = 0  # 未匹配左括號 '(' 的最多可能數量

        for c in s:
            if c == "(":
                left_min += 1
                left_max += 1
            elif c == ")":
                left_min -= 1
                left_max -= 1
            else:  # c == "*"
                left_min -= 1  # 把 '*' 當作 ')'
                left_max += 1  # 把 '*' 當作 '('

            # 若連把所有 '*' 都當成 '(' 仍不夠抵銷 ')'，直接無效
            if left_max < 0:
                return False

            # 未匹配左括號數量不能為負數（代表之前某些 '*' 選擇當成空字串即可）
            if left_min < 0:
                left_min = 0

        return left_min == 0
```


```Python
# 另一種直觀寫法：雙堆疊法（紀錄 '(' 與 '*' 的索引位置）
class Solution:
    def checkValidString(self, s: str) -> bool:
        left_stack = []
        star_stack = []

        for i, c in enumerate(s):
            if c == "(":
                left_stack.append(i)
            elif c == "*":
                star_stack.append(i)
            else:  # c == ")"
                if left_stack:
                    left_stack.pop()
                elif star_stack:
                    star_stack.pop()
                else:
                    return False

        # 處理剩餘未匹配的 '('，必須有出現在它右側的 '*' 來充當 ')'
        while left_stack and star_stack:
            if left_stack[-1] > star_stack[-1]:
                return False
            left_stack.pop()
            star_stack.pop()

        return len(left_stack) == 0
```

解釋: step1: 維護一個區間 `[left_min, left_max]`，代表遍歷到目前為止「尚未被抵銷的左括號 `(` 數量」的最小與最大可能值 step2: 以 `s = "(*))"` 為例，初始狀態 `left_min = 0, left_max = 0` step3: 讀取第 1 個字元 `'('`：確定多一個左括號，區間變為 `[1, 1]` step4: 讀取第 2 個字元 `'*'`：有三種選擇（當 `')'` 減 1、當 `""` 不變、當 `'('` 加 1），區間擴張為 `[0, 2]` step5: 讀取第 3 個字元 `')'`：確定抵銷一個左括號，區間平移為 `[-1, 1]`；因為 `left_min` 不能小於 0（代表剛剛的 `*` 不當 `')'` 而是當空字串），將 `left_min` 修正回 `0`，區間為 `[0, 1]` step6: 讀取第 4 個字元 `')'`：區間變為 `[-1, 0]`，同樣將 `left_min` 修正回 `0`，區間為 `[0, 0]` step7: 遍歷結束後檢查 `left_min == 0` 為 `True`，代表存在至少一種合法解釋讓所有括號完全抵銷，所以答案 = `True`

解釋: 為什麼可以用區間 `[left_min, left_max]` 來代表所有可能性？

因為每遇到一個 `*`，未匹配左括號的數量變化只會是 `-1`、`0` 或 `+1`，所有合法的狀態數量永遠會形成一個**連續的整數區間**。只要過程中最大可能值 `left_max` 不跌破 `0`（右括號沒有多到無解），且結束時最小可能值 `left_min` 能夠歸零（左括號沒有多到消不掉），就保證一定存在合法的配對方式。

### **LeetCode 678 / LintCode 1089 - Valid Parenthesis String（有效的括號字串）**

這道題目是 **貪心演算法（Greedy）與括號匹配（Parentheses Matching）** 的經典進階題（NeetCode 150 必考題）。在傳統括號驗證的基礎上加入了萬用字元 `*`，使每個位置產生了 3 種分支選擇，考驗如何將指數級的搜尋空間壓縮為線性時間的狀態追蹤。

## **解法解析**

這道題可以轉化為 **動態區間邊界追蹤（Range Tracking）** 來求解。

### **思路**

1. **觀察痛點**
    
    - 在沒有 `*` 的基本題（LeetCode 20）中，我們只需要用一個整數 `balance` 記錄未匹配的 `(` 數量：遇到 `(` 就 `+1`，遇到 `)` 就 `-1`，途中不能小於 `0`，最後必須等於 `0`。
        
    - 加入 `*` 後，每個 `*` 都有 `+1`、`-1`、`0` 三種分支，若直接暴力遞迴（DFS），時間複雜度會高達 $O(3^n)$。
        
2. **發現核心規律（連續可能性區間）**
    
    - 雖然分岐很多，但所有合法路徑的 `balance` 值在每一步都必定構成一個**連續區間 `[left_min, left_max]`**：
        
        - **`left_max`（最樂觀情況）**：把所有 `*` 都當成 `(` 來增加左括號儲備。如果連這樣都出現 `left_max < 0`，表示右括號 `)` 實在太多，後面再多左括號也救不回來，直接回傳 `False`。
            
        - **`left_min`（最嚴格情況）**：盡可能把 `*` 當成 `)` 去消耗左括號。但如果 `left_min < 0`，代表我們「太積極」把 `*` 當成右括號了，此時只需將這些 `*` 改視為空字串 `""`，即可把 `left_min` 維持在 `0`。
            
3. **代碼執行流程**
    
    - 初始化 `left_min = 0`、`left_max = 0`。
        
    - 單次遍歷字串 `s`：
        
        - 遇到 `(`：`left_min += 1`、`left_max += 1`
            
        - 遇到 `)`：`left_min -= 1`、`left_max -= 1`
            
        - 遇到 `*`：`left_min -= 1`、`left_max += 1`
            
        - 每一步檢查：若 `left_max < 0` 立即回傳 `False`；若 `left_min < 0` 則重置 `left_min = 0`。
            
    - 迴圈結束後，檢查 `left_min == 0`（確認所有強制存在的 `(` 都能被成功抵銷）。
        

## **具體舉例**

假設 `s = "(*))"`，遍歷過程中的區間變化如下：

|**索引 i**|**字元 c**|**原始變化 [min, max]**|**邊界修正後 [left_min, left_max]**|**狀態說明**|
|---|---|---|---|---|
|初始|-|`[0, 0]`|`[0, 0]`|尚未讀取任何字元|
|`0`|`'('`|`[1, 1]`|`[1, 1]`|必須有 1 個左括號待匹配|
|`1`|`'*'`|`[0, 2]`|`[0, 2]`|`*` 可當 `)`、`""` 或 `(`，可能數量為 `0, 1, 2`|
|`2`|`')'`|`[-1, 1]`|`[0, 1]`|`left_min < 0` 修正為 `0`（代表 `*` 不當 `)`）|
|`3`|`')'`|`[-1, 0]`|`[0, 0]`|`left_min < 0` 修正為 `0`，最終最小未匹配數為 `0`|

最後 `left_min == 0` 成立，返回 `True`。

## **時間與空間複雜度分析**

- **時間複雜度：$O(n)$**
    
    - 僅需對長度為 $n$ 的字串進行一次線性掃描，每次迭代執行常數時間的加減與比較，因此時間複雜度為 **$O(n)$**。
        
- **空間複雜度：$O(1)$**
    
    - 只使用了 `left_min` 和 `left_max` 兩個整數變數，不需要額外的陣列或堆疊，空間複雜度為 **$O(1)$**。
        

## **其他解法**

### **1. 雙堆疊法（Two Stacks）**

- **思路**：
    
    - 使用兩個堆疊 `left_stack` 和 `star_stack`，分別儲存尚未匹配的 `(` 和 `*` 的**索引位置**。
        
    - 遇到 `)` 時，優先從 `left_stack` 彈出左括號匹配（把 `*` 留到最後當萬用牌）；若 `left_stack` 為空，才從 `star_stack` 彈出 `*` 頂替。
        
    - 遍歷結束後，若 `left_stack` 還有剩餘的 `(`，必須與 `star_stack` 配對，且**星號的索引必須大於左括號的索引**（`star_stack[-1] > left_stack[-1]`），才能充當右括號。
        
- **時間複雜度**：$O(n)$
    
- **空間複雜度**：$O(n)$（最壞情況下需儲存所有字元的索引）
    

### **2. 動態規劃 / 記憶化搜尋（DP / Memoization）**

- **思路**：
    
    - 定義狀態 `dp[i][open_count]` 表示處理到第 `i` 個字元，且目前有 `open_count` 個未匹配左括號時，最終能否構成有效字串。
        
    - 遇到 `*` 時轉移到 `dp[i+1][open_count+1]`、`dp[i+1][open_count-1]` 或 `dp[i+1][open_count]`。
        
- **時間複雜度**：$O(n^2)$
    
- **空間複雜度**：$O(n^2)$（可透過滾動陣列優化至 $O(n)$）
    

## **總結**

- **面試最佳解（Greedy 區間法）**：時間 **$O(n)$**、空間 **$O(1)$**，代碼極為精簡，是面試官最期待看到的標準最優解。
    
- **次佳解（雙堆疊法）**：時間 **$O(n)$**、空間 **$O(n)$**，邏輯非常直觀，當面試官要求「同時輸出合法配對方式或替換後的字串」時，雙堆疊法因為保留了每個字元的實際索引位置，會比貪心法更容易擴展。
    

## **LeetCode / LintCode 括號匹配類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 20 / LintCode 423**|Valid Parentheses（有效的括號）|**多種類括號基本匹配 `()[]{}`**|**堆疊（Stack）** + 雜湊表對應左右括號，$O(n)$|
|**LeetCode 678 / LintCode 1089**|Valid Parenthesis String（有效的括號字串）|**單種括號 + 萬用字元 `*` 驗證**|**貪心區間 `[left_min, left_max]`**，$O(n)$ 時間 $O(1)$ 空間|
|**LeetCode 2116**|Check if a Parentheses String Can Be Valid|**給定 `locked` 陣列，未鎖定位置可任意改為 `(` 或 `)`**|**雙向貪心掃描（從左到右 + 從右到左）** 或區間法|
|**LeetCode 22 / LintCode 427**|Generate Parentheses（括號生成）|**生成所有 `n` 對合法括號組合**|**回溯法（Backtracking）**，限制 `left < n` 且 `right < left`|
|**LeetCode 32 / LintCode 193**|Longest Valid Parentheses（最長有效括號）|**求最長連續有效括號子字串長度**|**堆疊存索引**、**1D 動態規劃** 或 **雙向計數器掃描**|
|**LeetCode 921**|Minimum Add to Make Parentheses Valid|**求最少需要插入幾個括號才能變有效**|**貪心計數器**，統計未匹配的 `(` 與多餘的 `)` 總和|
|**LeetCode 1249**|Minimum Remove to Make Valid Parentheses|**刪除最少無效括號並回傳合法字串**|**堆疊紀錄無效括號索引** + `set` 過濾重建字串|
|**LeetCode 301 / LintCode 780**|Remove Invalid Parentheses（刪除無效的括號）|**刪除最少括號，列出所有可能的合法結果**|**BFS（最短路徑）** 或 **計算多餘左右括號數後回溯剪枝**|

### **進階變形題：如果字串包含多種括號 `()`, `[]`, `{}` 以及萬用字元 `*`，該如何判斷？**

當出現多種括號（例如 `"([*)]"`）時，由於括號必須遵守**後進先出（LIFO）的巢狀順序**，單純計數的貪心區間法會失效（無法區分 `(` 與 `[` 的順序），此時必須改用 **記憶化搜尋（DFS + Memoization）** 來追蹤當前尚未閉合的括號堆疊狀態：

Python

```
from functools import lru_cache

class Solution:
    def checkValidMultiBrackets(self, s: str) -> bool:
        pairs = {")": "(", "]": "[", "}": "{"}
        openers = ("(", "[", "{")

        @lru_cache(None)
        def dfs(i: int, stack: tuple[str, ...]) -> bool:
            # 剪枝：剩餘字元數不足以清空目前堆疊
            if len(stack) > len(s) - i:
                return False
            if i == len(s):
                return len(stack) == 0

            c = s[i]
            if c in openers:
                return dfs(i + 1, stack + (c,))
            elif c in pairs:
                if not stack or stack[-1] != pairs[c]:
                    return False
                return dfs(i + 1, stack[:-1])
            else:  # c == "*"
                # 1. 當作空字串
                if dfs(i + 1, stack):
                    return True
                # 2. 當作右括號（彈出堆疊頂端）
                if stack and dfs(i + 1, stack[:-1]):
                    return True
                # 3. 當作任意一種左括號
                for op in openers:
                    if dfs(i + 1, stack + (op,)):
                        return True
                return False

        return dfs(0, ())
```