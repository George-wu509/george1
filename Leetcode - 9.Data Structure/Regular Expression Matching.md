Lintcode 154
实现支持 `'.'` 和 `'*'` 的正则表达式匹配。`'.'` 匹配任意一个字母。`'*'` 匹配零个或者多个前面的元素，`'*'` 前保证是一个非 `'*'` 元素。匹配应该覆盖整个输入字符串，而不仅仅是一部分。需要实现的函数是：  
bool isMatch(string s, string p)

**样例 1:**
```python
"""
输入："aa"，"a"
输出：false
解释：
无法匹配
```
**样例 2:**
```python
"""
输入："aa"，"a*"
输出：true
解释：
'*' 可以重复 a
```
**样例 3:**
```python
"""
输入："aab", "c*a*b"
输出：true
解释：
"c*" 作为一个整体匹配 0 个 'c' 也就是 ""
"a*" 作为一个整体匹配 2 个 'a' 也就是 "aa"
"b" 匹配 "b"
所以 "c*a*b" 可以匹配 "aab"
```
**样例4：**
```python
"""
输入："abcc", ".*"
输出：true
解释：
".*" 作为一个整体匹配 4 个 '.' 也就是 "...."
第一个 "." 匹配第一个字符 "a"
第二个 "." 匹配第二个字符 "b"
最后两个 "." 匹配字符 "cc"
```


```python
class Solution:
    """
    @param s: A string 
    @param p: A string includes "?" and "*"
    @return: is Match?
    """
    def is_match(self, s, p):
        return self.is_match_helper(s, 0, p, 0, {})
        
        
    # s 从 i 开始的后缀能否匹配上 p 从 j 开始的后缀
    # 能 return True
    def is_match_helper(self, s, i, p, j, memo):
        if (i, j) in memo:
            return memo[(i, j)]
        
        # s is empty
        if len(s) == i:
            return self.is_empty(p[j:])
            
        if len(p) == j:
            return False
            
        if j + 1 < len(p) and p[j + 1] == '*':
            matched = self.is_match_char(s[i], p[j]) and self.is_match_helper(s, i + 1, p, j, memo) or \
                self.is_match_helper(s, i, p, j + 2, memo)
        else:                
            matched = self.is_match_char(s[i], p[j]) and self.is_match_helper(s, i + 1, p, j + 1, memo)
        
        memo[(i, j)] = matched
        return matched
        
        
    def is_match_char(self, s, p):
        return s == p or p == '.'
        
    def is_empty(self, p):
        if len(p) % 2 == 1:
            return False
        
        for i in range(len(p) // 2):
            if p[i * 2 + 1] != '*':
                return False
        return True
```
pass


# **LintCode 154: Regular Expression Matching 解法分析**

## **解題目標**

給定一個字串 `s` 和一個模式 `p`，其中 `p` 可能包含：

- `.`：匹配任意一個字元。
- `*`：表示 `*` 前面的字符可以出現 **0 次或多次**。

判斷 `s` 是否能被 `p` 完全匹配。

---

## **解法核心**

這題是經典的 **正則表達式匹配 (Regex Matching)** 問題，核心是：

1. **遞迴 (Recursion) + 記憶化 (Memoization)**
2. **動態規劃 (Dynamic Programming)**

### **為什麼要用遞迴？**

- **規則非常複雜**，但可以拆解成 **子問題 (subproblems)**，即：
    
    - `p[j] == s[i]` 或 `p[j] == '.'` → 繼續匹配 `s[i+1]` 和 `p[j+1]`
    - `p[j] == '*'` → 可能選擇 `0` 次 (`跳過 p[j-1]`) 或 `多次` (`繼續匹配 s[i]`)
- **重複子問題**：相同 `s[i:]` 和 `p[j:]` 可能被多次計算，所以用 **記憶化 (Memoization)** 來避免 **指數時間複雜度**。
    

---

## **解法步驟**

### **Step 1: 建立遞迴函數 `is_match_helper(s, i, p, j, memo)`**

- **目標**：判斷 `s[i:]` 是否能匹配 `p[j:]`。
- **備忘錄 `memo`**：記錄 `(i, j)` 的匹配結果，避免重複計算。

---

### **Step 2: 基本情況 (Base Cases)**

1. **字串 `s` 已經匹配完 (`i == len(s)`)**
    
    - 這時候 `p[j:]` 可能還有剩下的 `*` 模式，比如 `"a*"`，所以要檢查：

        `if len(s) == i:     return self.is_empty(p[j:])`
        
    - **`is_empty(p[j:])` 方法**
        - 確保 `p[j:]` 的每個字符後面都跟著 `*`，否則 `s` 無法匹配 `p`。
2. **模式 `p` 已經匹配完 (`j == len(p)`)**
    
    - 若 `s` 仍有未匹配的字符，則返回 `False`。

---

### **Step 3: 遞迴處理 `p[j]`**

1. **`p[j + 1] == '*'` (星號匹配)**
    
    - `'*'` 可以讓 `p[j]` **匹配 0 次或多次**：
        
        python
        
        複製編輯
        
        `if j + 1 < len(p) and p[j + 1] == '*':     matched = (self.is_match_char(s[i], p[j]) and self.is_match_helper(s, i + 1, p, j, memo)) or \               self.is_match_helper(s, i, p, j + 2, memo)`
        
    - **兩種選擇**：
        - **使用 `p[j]` (`s[i]` 和 `p[j]` 匹配成功，則繼續 `s[i+1]` 對 `p[j]`)**。
        - **不使用 `p[j]` (`跳過` `p[j:j+2]`，因為 `*` 可以讓 `p[j]` 出現 0 次)**。
2. **`p[j] != '*'` (正常匹配)**
    
    - 如果 `p[j]` 不是 `'*'`，則直接匹配當前字符：
        
        python
        
        複製編輯
        
        `matched = self.is_match_char(s[i], p[j]) and self.is_match_helper(s, i + 1, p, j + 1, memo)`
        
    - **`is_match_char(s, p)` 方法**
        
        python
        
        複製編輯
        
        `def is_match_char(self, s, p):     return s == p or p == '.'`
        
    - **兩種可能匹配成功的情況**：
        - `s[i] == p[j]`（字符完全匹配）
        - `p[j] == '.'`（可以匹配任何字符）

---

### **Step 4: 記憶化遞迴**

- **為了避免重複計算 `(i, j)`，我們將結果存入 `memo`**
- 若已經計算過 `(i, j)`，直接返回：

    `if (i, j) in memo:     return memo[(i, j)]`
    

---

## **變數定義**

|變數名稱|作用|
|---|---|
|`s`|目標字串|
|`p`|正則表達式模式|
|`i`|`s` 的當前索引|
|`j`|`p` 的當前索引|
|`memo`|記錄 `(i, j)` 的匹配結果，避免重複計算|
|`matched`|當前匹配結果|
|`is_match_char(s[i], p[j])`|判斷 `s[i]` 是否匹配 `p[j]`|
|`is_empty(p[j:])`|判斷 `p[j:]` 是否為空（只包含 `a*`、`b*` 類型）|

---

## **具體範例**

### **範例 1**

`輸入: s = "aa", p = "a*"`

**匹配過程**

1. `p[1] == '*'`：
    - `'*'` 代表 `p[0] = 'a'` **可以匹配 0 次或多次**。
    - **選擇匹配 0 次** → `s = "aa", p = ""`（匹配 `p[j+2:]`）
    - **選擇匹配 1 次** → `s = "a", p = "a*"`
    - **選擇匹配 2 次** → `s = "", p = "a*"`

**結果**

`True`

---

### **範例 2**

`輸入: s = "mississippi", p = "mis*is*p*."`

**匹配過程**

- `"mis*"` → `"m"` `"i"` `"s*"` (`s*` 可匹配 `s` 0 或多次)
- `"is*p*"` → `"i"` `"s*"` `"p*"`
- `.` 可匹配 `i`
- 最終匹配成功

**結果**

`True`

---

## **時間與空間複雜度分析**

### **時間複雜度**

- **最差情況**：每個 `(i, j)` 都可能被計算一次。
- **透過 `memo` 優化後**，總共有 `O(m * n)` 個可能的 `(i, j)` 狀態（`m` 為 `s` 長度，`n` 為 `p` 長度）。
- **時間複雜度：`O(m * n)`**。

### **空間複雜度**

- **遞迴深度** 最多 `O(m + n)`。
- **備忘錄 (`memo`)** 需要 `O(m * n)` 空間。
- **總計：`O(m * n)`**。

---

## **其他解法 (不寫 Code)**

1. **動態規劃 `O(m * n)`**
    
    - `dp[i][j]` 表示 `s[:i]` 是否能匹配 `p[:j]`。
    - 使用 `dp` 陣列存儲所有可能狀態，避免遞迴。
2. **暴力遞迴 `O(2^m)`**
    
    - 直接嘗試所有可能的 `*` 匹配方式，時間複雜度 **指數級別**，不可行。

---

## **總結**

|**解法**|**時間複雜度**|**適用場景**|**優缺點**|
|---|---|---|---|
|**遞迴 + 記憶化 (`O(m * n)`)**|`O(m * n)`|最適合一般情境|✅ 高效，可處理大 `m, n`|
|**動態規劃 (`O(m * n)`)**|`O(m * n)`|適合大量模式匹配|✅ 記憶化優化，無遞迴|
|**暴力遞迴 (`O(2^m)`)**|`O(2^m)`|適用於小數據|❌ 太慢，不適用|

✅ **最佳選擇：遞迴 + 記憶化 (`O(m * n)`)**，時間 `O(m * n)`，空間 `O(m * n)`，適合所有場景 🚀

  

O