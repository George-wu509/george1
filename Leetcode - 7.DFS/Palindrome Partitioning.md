
LeetCode 131 / LintCode 136 (NeetCode 150)
給你一個字串 `s` ，請你將 `s` 分割成一些子字串，使每個子字串都是 **回文串（Palindrome）** 。返回 `s` 所有可能的分割方案。
**回文串** 是正著讀和反著讀都一樣的字串（例如 `"a"`、`"aba"`、`"abba"`）。

**樣例 1：**
輸入：
```python
s = "aab"
```
輸出：
```python
[["a", "a", "b"], ["aa", "b"]]
```
解釋：
共有兩種合法切割方式：
1. 切為 `"a"`、`"a"`、`"b"`，三段皆為長度 1 的回文串。
2. 切為 `"aa"`、`"b"`，兩段皆為回文串。
（若切為 `"a"`、`"ab"`，因 `"ab"` 不是回文串，故不合法）

**樣例 2：**
輸入：
```python
s = "a"
```
輸出：
```python
[["a"]]
```
解釋：
單一字元本身即為回文串，只有一種分割方案

**樣例 3：**
輸入：
```python
s = "aba"
```
輸出：
```python
[["a", "b", "a"], ["aba"]]
```
解釋：
可以逐字切開成三個單字元，也可以整段 `"aba"` 不切分直接作為一個完整回文串

---

### 解法一：標準回溯法 + 雙指標判斷回文（NeetCode 經典寫法，空間 O(n)）

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        part = []

        def dfs(i: int) -> None:
            # 1. 切割線已經走到字串最尾端，代表整條路徑的所有子字串皆為回文
            if i >= len(s):
                res.append(part.copy())
                return

            # 2. 枚舉當前這一刀要切在哪個結束位置 j（子字串為 s[i : j + 1]）
            for j in range(i, len(s)):
                if self.isPali(s, i, j):
                    # 做選擇：將合法的回文子字串加入當前分割路徑
                    part.append(s[i : j + 1])
                    # 遞迴處理剩餘未切割的後綴 s[j + 1 :]
                    dfs(j + 1)
                    # 撤銷選擇（回溯）
                    part.pop()

        dfs(0)
        return res

    def isPali(self, s: str, l: int, r: int) -> bool:
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True
```
pass

### 解法二：回溯法 + 動態規劃預處理回文表（面試最佳解，避免重複驗證回文）

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        # 1. 動態規劃預處理：dp[i][j] 表示子字串 s[i..j] 是否為回文串
        dp = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True

        res = []
        path = []

        # 2. 結合 O(1) 查表進行回溯搜尋
        def backtrack(start: int) -> None:
            if start == n:
                res.append(path.copy())
                return

            for end in range(start, n):
                # 直接 O(1) 查詢 dp[start][end]，省去雙指標重複比對
                if dp[start][end]:
                    path.append(s[start : end + 1])
                    backtrack(end + 1)
                    path.pop()

        backtrack(0)
        return res
```

解釋:
step1: 先建立一張 `n x n` 的二維布林表 `dp`，從後往前填表（`i` 從 `n - 1` 到 `0`），只要頭尾字元相同 `s[i] == s[j]`，且長度小於等於 3（`j - i <= 2`）或內部子字串 `dp[i + 1][j - 1]` 為回文，就將 `dp[i][j]` 設為 `True`。
step2: 假設 `s = "aab"`（長度 `n = 3`），預處理完畢後：`dp[0][0] = True ("a")`、`dp[0][1] = True ("aa")`、`dp[1][1] = True ("a")`、`dp[2][2] = True ("b")`，其餘為 `False`。
step3: 呼叫 `backtrack(start = 0)`，開始枚舉第一刀的結束位置 `end`：
  - `end = 0`：查詢 `dp[0][0] == True`，切出 `"a"`，`path = ["a"]`，遞迴呼叫 `backtrack(1)`。
step4: 進入 `backtrack(start = 1)`（處理剩餘字串 `"ab"`）：
  - `end = 1`：查詢 `dp[1][1] == True`，切出 `"a"`，`path = ["a", "a"]`，遞迴呼叫 `backtrack(2)`。
  - 在 `backtrack(2)` 中，`end = 2` 時 `dp[2][2] == True`，切出 `"b"`，`path = ["a", "a", "b"]`，呼叫 `backtrack(3)` 觸發 `start == n`，**存入第一個答案 `["a", "a", "b"]`**！
  - 回溯彈出 `"b"` 與第二個 `"a"`，回到 `backtrack(1)` 繼續嘗試 `end = 2`（子字串 `"ab"`）：查詢 `dp[1][2] == False`，不合法直接跳過！
step5: 回到最外層 `backtrack(start = 0)`（`path` 恢復為 `[]`），繼續枚舉下一刀：
  - `end = 1`：查詢 `dp[0][1] == True`，切出 `"aa"`，`path = ["aa"]`，遞迴呼叫 `backtrack(2)`。
  - 在 `backtrack(2)` 中，`end = 2` 切出 `"b"`，`path = ["aa", "b"]`，呼叫 `backtrack(3)` 觸發 `start == n`，**存入第二個答案 `["aa", "b"]`**！
step6: 回到 `backtrack(0)` 嘗試 `end = 2`（子字串 `"aab"`）：查詢 `dp[0][2] == False` 跳過，迴圈結束，最終回傳 `[["a", "a", "b"], ["aa", "b"]]`。

---

### **LeetCode 131 - Palindrome Partitioning（分割回文串）**

這道題目是 **回溯演算法（Backtracking）** 在字串切割問題上的標準代表題，與「組合總和（Combination Sum）」在搜尋樹結構上完全同構——差別只在於本題每次選取的不是陣列數字，而是 **字串的一段連續前綴**。

---

## **解法解析**

寫好「字串切割類回溯題」的關鍵，在於把「切刀位置」抽象成一棵多叉搜尋樹。

### **思路**

1. **切割問題就是組合問題（樹形抽象）**
   - 對於長度為 `n` 的字串 `s`，字元之間共有 `n - 1` 個可以下刀的縫隙（以及字串最尾端 `n`）。
   - 我們用參數 `start`（或 `i`）表示 **當前剩餘未切割字串的起始索引**。在每一層遞迴中，用 `for end in range(start, n)` 嘗試把 `s[start : end + 1]` 作為當前切下來的第一段：
     - **合法性剪枝**：只有當 `s[start : end + 1]` 本身是回文串時，這條分支才值得繼續往下切；若連當前這段都不是回文，直接跳過（剪枝），絕不浪費時間遞迴後面的字元。
     - **終止條件**：當 `start == n` 時，代表整個字串已經從頭到尾被完美切成若干段回文子字串，將 `path.copy()` 加入最終結果 `res`。
2. **為什麼要搭配 DP 預處理回文表？**
   - 在解法一中，每次遇到 `s[start : end + 1]` 都要呼叫一次 `isPali()` 花費 `O(n)` 時間檢查；但在回溯過程中，同一個區間 `[start, end]` 會在不同分支裡被反覆檢查很多次。
   - 透過事先花 `O(n²)` 時間建立 `dp[start][end]` 布林矩陣，之後在回溯樹的每一個節點都能用 **`O(1)` 時間** 瞬間判斷子字串是否為回文，消除重複計算。
3. **DP 填表順序細節**
   - 因為狀態轉移方程 `dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]` 依賴於 **左下角** 的狀態 `(i + 1, j - 1)`，所以外層迴圈 `i` 必須 **由大到小（從 `n - 1` 遞減至 `0`）**，內層迴圈 `j` 則 **由小到大（從 `i` 遞增至 `n - 1`）**。

---

## **具體舉例**

以 `s = "aab"`（`n = 3`）為例，觀察回溯樹每一步的切割與剪枝狀態：

| 遞迴深度 | `start` 索引 | 嘗試結束 `end` | 截取子字串 `s[start:end+1]` | 是否回文 | 當前路徑 `path` | 執行動作與說明 |
|---|---|---|---|---|---|---|
| 0 | `0` | `0` | `"a"` | `True` | `["a"]` | 合法，進入下一層 `start = 1` |
| 1 | `1` | `1` | `"a"` | `True` | `["a", "a"]` | 合法，進入下一層 `start = 2` |
| 2 | `2` | `2` | `"b"` | `True` | `["a", "a", "b"]` | 合法，進入 `start = 3 == n` -> **收集 `["a", "a", "b"]`** |
| 1 | `1` | `2` | `"ab"` | `False` | `["a"]` | 回溯後改切 `"ab"`，非回文 **剪枝跳過** |
| 0 | `0` | `1` | `"aa"` | `True` | `["aa"]` | 回溯至最外層改切 `"aa"`，合法，進入 `start = 2` |
| 1 | `2` | `2` | `"b"` | `True` | `["aa", "b"]` | 合法，進入 `start = 3 == n` -> **收集 `["aa", "b"]`** |
| 0 | `0` | `2` | `"aab"` | `False` | `[]` | 回溯至最外層改切 `"aab"`，非回文 **剪枝跳過** |

---

## **時間與空間複雜度分析**

設字串 `s` 的長度為 `N`。

- **時間複雜度：O(N × 2^N)**
  - 最壞情況下（例如 `s = "aaaa"`，所有字元完全相同），任意子字串都是回文串。字串中有 `N - 1` 個縫隙，每個縫隙都可以選擇「切」或「不切」，因此總共有 `2^(N - 1)` 種合法分割方案。
  - 每找到一種完整方案，都需要花費 `O(N)` 的時間將 `path` 複製（`path.copy()`）並構建子字串存入 `res`。
  - 解法二雖然用 `O(N²)` 的 DP 預處理把單次回文檢查降為 `O(1)`，但輸出所有方案本身的漸進上界仍由組合數量與複製開銷主導，因此總時間複雜度為 **O(N × 2^N)**。
- **空間複雜度：O(N²)**（解法二）/ **O(N)**（解法一）
  - 不計算儲存最終答案 `res` 的輸出空間：
  - **解法一**：僅使用遞迴呼叫堆疊與暫存列表 `path`，深度最多為 `N`，空間複雜度為 **O(N)**。
  - **解法二**：額外維護了一張 `N × N` 的二維 DP 布林表，因此空間複雜度為 **O(N²)**（由於題目限制通常 `N <= 16`，`16 × 16` 的陣列微乎其微，換取極高的執行速度非常划算）。

---

## **其他解法**

### **1. 回溯 + 記憶化搜尋（Backtracking with Memoization）**
- **思路**：
  - 定義 `dfs(start)` 直接回傳「從索引 `start` 到字串結尾的所有合法分割方案列表」，並用雜湊表 `memo[start]` 快取計算過的結果。
  - 當不同前綴切割方式（例如 `"a" + "a"` 與 `"aa"`）都走到同一個後綴起點 `start = 2` 時，可以直接從 `memo[2]` 取出已算好的所有後綴組合，避免重複展開相同的後半段子樹。
- **時間複雜度**：O(N × 2^N)（消除重複子問題，在有大量局部重複回文的字串上表現極佳）
- **空間複雜度**：O(N × 2^N)（需快取中間各個後綴起點的所有分割結果）

---

## **總結**

- **面試首選寫法**：**解法一（標準回溯 + 雙指標 `isPali`）**，程式碼結構最乾淨，空間 `O(N)`，在面試白板題中不容易寫錯 DP 填表索引，足以輕鬆通過所有測試。
- **進階優化加分項**：向面試官點出「雙指標會對相同區間重複驗證回文」，並給出 **解法二（`O(N²)` DP 預處理回文表 + 回溯）**，展現對時間與空間權衡（Time-Space Tradeoff）的掌握度。

---

## **回文串 / 字串分割回溯類問題整理**

| 題號 | 題目名稱 | 問題類型 | 解法 |
|---|---|---|---|
| **LeetCode 131** | Palindrome Partitioning（分割回文串） | 找出將字串 `s` 分割成全為回文子字串的 **所有方案** | **回溯法 + 雙指標 / DP 預處理回文表**（時間 O(N · 2^N)） |
| **LeetCode 132** | Palindrome Partitioning II（分割回文串 II） | 求將字串 `s` 分割成全為回文子字串的 **最少切割次數** | **兩階段動態規劃**：先 `O(N²)` 算回文表，再用 1D DP `cuts[i]` 求最小切刀數（時間 O(N²)） |
| **LeetCode 647** | Palindromic Substrings（回文子串） | 統計字串 `s` 中總共有多少個回文子字串 | **中心擴展法（Expand Around Center）** 或 2D DP（時間 O(N²)，空間 O(1)） |
| **LeetCode 5** | Longest Palindromic Substring（最長回文子串） | 找出字串 `s` 中最長的一個回文子字串 | **中心擴展法**：枚舉 `2N - 1` 個中心向兩側擴展（時間 O(N²)，空間 O(1)） |
| **LeetCode 93** | Restore IP Addresses（復原 IP 位址） | 將數字字串切割成 4 段合法的 IPv4 位址（每段 `0 ~ 255` 且無前導零） | **字串切割回溯法**：與 LeetCode 131 同模板，增加「已切滿 4 段」與「數值合法性」剪枝 |
| **LeetCode 139** | Word Break（單詞拆分） | 判斷字串 `s` 是否能被空格拆分為字典 `wordDict` 中的單詞 | **1D 動態規劃 / BFS**（只需判斷 `True/False`，不需列出所有路徑，時間 O(N²)） |
| **LeetCode 140** | Word Break II（單詞拆分 II） | 列出字串 `s` 拆分為字典單詞的 **所有可能句子** | **記憶化回溯（DFS + Memoization）**（與 LeetCode 131 同屬「列舉所有合法切割方案」） |

---

### **進階變形題：如果只問「最少需要切幾刀」才能讓每段都是回文（LeetCode 132 Palindrome Partitioning II）？**

當題目從「列出所有方案」變成「求 **最少** 切割次數（最值問題）」時，就不再使用指數級的回溯法，而是升級為純 **動態規劃（`O(N²)` 時間）**：先用與解法二相同的 `is_pal[j][i]` 判斷區間是否回文，再定義 `dp[i]` 為前綴 `s[0..i]` 的最少切割次數：

```python
class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        # 1. is_pal[j][i] 記錄 s[j..i] 是否為回文串
        is_pal = [[False] * n for _ in range(n)]
        # 2. dp[i] 記錄 s[0..i] 最少需要切幾刀（最壞情況每個字元都切開，需要 i 刀）
        dp = [i for i in range(n)]

        for i in range(n):
            for j in range(i + 1):
                if s[j] == s[i] and (i - j <= 2 or is_pal[j + 1][i - 1]):
                    is_pal[j][i] = True
                    # 若從開頭 0 到 i 整段就是回文，則完全不用切（0 刀）
                    if j == 0:
                        dp[i] = 0
                    # 否則在 j - 1 與 j 之間切一刀，轉移自 dp[j - 1] + 1
                    else:
                        dp[i] = min(dp[i], dp[j - 1] + 1)

        return dp[n - 1]
```
````