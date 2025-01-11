
### **LintCode 582：Word Break II**

#### **題目描述**

給定一個字符串 `s` 和一個單詞字典 `wordDict`，在確保字符串可以完全被字典中的單詞分割的前提下，返回所有可能的分割方式。

例如：
輸入: 
s = "catsanddog", 
wordDict = ["cat", "cats", "and", "sand", "dog"]
輸出: 
["cats and dog", "cat sand dog"]

---

### **DFS 解法**

#### **算法思路**

使用深度優先搜索（DFS）枚舉所有可能的分割方式，並通過緩存進行剪枝來減少重複計算。

1. **檢查可行性**：
    
    - 使用動態規劃 `dp` 判斷是否可以將 `s` 完全分割成字典中的單詞。如果無法分割，直接返回空列表。
2. **DFS 遍歷**：
    
    - 從字符串的起始位置開始，嘗試匹配字典中的單詞。
    - 若匹配成功，將當前單詞加入結果，遞歸處理剩餘部分。
    - 使用緩存記錄每個起始位置的結果，避免重複計算。
3. **結果組合**：
    
    - 每次遞歸結束時，返回當前部分分割的所有可能，最終合併形成完整的結果。

---
Example:
**样例 1:**
```
输入："lintcode"，["de","ding","co","code","lint"]
输出：["lint code", "lint co de"]
解释：
插入一个空格是"lint code"，插入两个空格是 "lint co de"
```
**样例 2:**
```
输入："a"，[]
输出：[]
解释：字典为空
```


#### **代碼詳解**

```python
class Solution:
    def word_break(self, s: str, word_dict: Set[str]) -> List[str]:
        wordSet = set(word_dict)  # 將字典轉為集合，加速查詢
        memo = {}  # 緩存每個起始位置的結果

        def dfs(start):
            if start in memo:  # 若已計算過，直接返回緩存結果
                return memo[start]
            
            if start == len(s):  # 當到達字符串末尾時，返回空字符串作為一種分割方式
                return [""]

            res = []
            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                if word in wordSet:  # 如果當前子字符串是字典中的單詞
                    # 獲取剩餘部分的分割結果
                    for sub in dfs(end):
                        if sub:
                            res.append(word + " " + sub)
                        else:
                            res.append(word)
            memo[start] = res  # 緩存結果
            return res

        return dfs(0)
```
pass

---

#### **執行過程舉例**

輸入：

`s = "catsanddog" wordDict = ["cat", "cats", "and", "sand", "dog"]`

1. **初始調用**：
    
    - `dfs(0)`，處理整個字符串。
    - 初始 `memo = {}`。
2. **第一層遞歸**：
    
    - 嘗試匹配從 `0` 開始的單詞：
        - 匹配 `cat`：
            - 遞歸調用 `dfs(3)`，處理剩餘字符串 `"sanddog"`。
        - 匹配 `cats`：
            - 遞歸調用 `dfs(4)`，處理剩餘字符串 `"anddog"`。
3. **第二層遞歸**：
    
    - 處理 `"sanddog"`：
        - 匹配 `sand`：
            - 遞歸調用 `dfs(7)`，處理剩餘字符串 `"dog"`。
    - 處理 `"anddog"`：
        - 匹配 `and`：
            - 遞歸調用 `dfs(7)`，處理剩餘字符串 `"dog"`。
4. **第三層遞歸**：
    
    - 處理 `"dog"`：
        - 匹配 `dog`，返回 `["dog"]`。
    - 回溯將結果合併：
        - `"sanddog"` 返回 `["sand dog"]`。
        - `"anddog"` 返回 `["and dog"]`。
5. **回溯組合**：
    
    - `"catsanddog"` 返回 `["cats and dog", "cat sand dog"]`。

---

### **複雜度分析**

1. **時間複雜度**：
    
    - 在最壞情況下，所有可能的分割方式會被枚舉。假設字典中單詞的平均長度為 mmm，則字符串的長度為 nnn，最壞情況下有 2n2^n2n 種分割方式。
    - 使用緩存減少重複計算，平均時間複雜度為 **O(2n)O(2^n)O(2n)**。
2. **空間複雜度**：
    
    - 遞歸調用棧的深度最多為字符串的長度 nnn。
    - 緩存和結果佔用的空間取決於分割方式的數量，最壞情況下為 **O(2n)O(2^n)O(2n)**。

---

### **其他解法簡述**

#### 1. **BFS（廣度優先搜索）**

- 使用佇列存儲每個分割點，逐步檢查可能的分割方式。
- 當到達字符串末尾時，將當前分割加入結果。
- 與 DFS 相比，BFS 可能更容易理解，但需要更多的內存存儲中間狀態。

#### 2. **動態規劃 + 路徑記錄**

- 使用 DP 判斷是否可以分割，並記錄每個位置的可能單詞。
- 最後通過回溯構造所有可能的分割方式。

```python
class Solution:
    def wordBreak(self, s, wordDict):
        wordSet = set(wordDict)
        n = len(s)
        dp = [[] for _ in range(n + 1)]  # dp[i] 存儲從 0 到 i 的所有分割方式
        dp[0] = [""]

        for i in range(1, n + 1):
            for j in range(i):
                if s[j:i] in wordSet:
                    for prev in dp[j]:
                        dp[i].append((prev + " " + s[j:i]).strip())

        return dp[-1]

```

- **時間複雜度**：O(n2)O(n^2)O(n2) 檢查所有可能的分割。
- **空間複雜度**：O(n2)O(n^2)O(n2)。