Lintcode 1086
给定两个字符串A和B，找到A必须重复的最小次数，以使得B是它的子字符串。 如果没有这样的解决方案，返回-1。

**样例1:**
```python
"""
输入 : A = "a"     B = "b".
输出 : -1
```
**样例 2:**
```python
"""
输入 : A = "abcd"     B = "cdabcdab".
输出 :3
解释：因为将A重复3次以后 (“abcdabcdabcd”), B将成为其的一个子串 ; 而如果A只重复两次 ("abcdabcd")，B并非其的一个子串.
```


```python
class Solution:
    def strstr(self, haystack: str, needle: str) -> int:
        n, m = len(haystack), len(needle)
        if m == 0:
            return 0

        k1 = 10 ** 9 + 7
        k2 = 1337
        mod1 = random.randrange(k1) + k1
        mod2 = random.randrange(k2) + k2

        hash_needle = 0
        for c in needle:
            hash_needle = (hash_needle * mod2 + ord(c)) % mod1
        hash_haystack = 0
        for i in range(m - 1):
            hash_haystack = (hash_haystack * mod2 + ord(haystack[i % n])) % mod1
        extra = pow(mod2, m - 1, mod1)
        for i in range(m - 1, n + m - 1):
            hash_haystack = (hash_haystack * mod2 + ord(haystack[i % n])) % mod1
            if hash_haystack == hash_needle:
                return i - m + 1
            hash_haystack = (hash_haystack - extra * ord(haystack[(i - m + 1) % n])) % mod1
            hash_haystack = (hash_haystack + mod1) % mod1
        return -1

    def repeated_string_match(self, a: str, b: str) -> int:
        # write your code here
        n, m = len(a), len(b)
        index = self.strstr(a, b)
        if index == -1:
            return -1
        if n - index >= m:
            return 1
        return (m + index - n - 1) // n + 2
```
pass


本題 **`repeated_string_match(self, a: str, b: str) -> int`** 的目標是找出，最少要將 `a` 重複幾次，才能讓 `b` 成為 `a` 的子字串。

### **核心觀察**

1. **最小重複次數界限**
    
    - `a` 最少需要重複 ⌈`len(b) / len(a)`⌉ 次，才能完全覆蓋 `b` 的長度。
    - 但因為 `b` 可能跨越 `a` 的尾部和頭部，因此最多要重複 **1 次額外的 `a`** 才能確認。
2. **子字串搜尋**
    
    - 需要檢查 `b` 是否是 `a` 重複若干次後的**子字串** (`substring`)。
3. **使用 Rabin-Karp 字符串哈希加速匹配**
    
    - 計算 `b` 的哈希值，並在 `a` 重複後的字串中 **使用滾動哈希進行搜索**。

---

## **變數表**

|變數名稱|含義|
|---|---|
|`a, b`|兩個輸入字符串|
|`n, m`|`a` 和 `b` 的長度|
|`index`|`b` 在 `a` 重複後的索引位置（使用 `strstr` 搜索）|
|`mod1, mod2`|Rabin-Karp 滾動哈希的兩個大質數|
|`hash_needle`|`b` (needle) 的哈希值|
|`hash_haystack`|`a` (haystack) 當前滑動窗口的哈希值|
|`extra`|`mod2^(m-1) % mod1`，用於移除最左側字符的影響|

---

## **具體步驟**

### **Step 1: 用 Rabin-Karp 計算 `b` 的哈希值**

- 逐個字符計算 `b` (`needle`) 的哈希值 `hash_needle`。

### **Step 2: 使用滾動哈希在 `a` 重複後的字串中搜尋 `b`**

- 計算 `a` 的前 `m-1` 個字符的哈希值 `hash_haystack`。
- 透過 **滑動窗口** 移動 `hash_haystack`，檢查是否匹配 `hash_needle`。
- 若匹配成功，則返回索引 `index`。

### **Step 3: 計算 `a` 需要重複的次數**

- **如果 `index == -1`**，則 `b` 無法出現在 `a` 重複後的字串中，返回 `-1`。
- 否則，根據 `index` 計算 `a` 需要重複的次數：
    - 若 `n - index >= m`，說明 `b` 已經完全包含在 `a` 本身中，只需 `1` 次。
    - 否則計算 `(m + index - n - 1) // n + 2`，確保 `b` 可以被完全覆蓋。

---

## **範例解析**

### **範例 1**

python

複製編輯

`a = "abcd" b = "cdabcdab"`

#### **Step 1: 滾動哈希搜尋 `b`**

- `b = "cdabcdab"` 的哈希值：計算 `hash_needle`
- `a + a + a = "abcdabcdabcd"`，檢查 `b` 是否存在

|`i`|`hash_haystack`|是否匹配 `hash_needle`|
|---|---|---|
|0|不匹配|❌|
|1|不匹配|❌|
|2|匹配 `"cdabcdab"`|✅|

- `index = 2`

#### **Step 2: 計算 `a` 需要重複的次數**

- `len(b) = 7`，`len(a) = 4`
- `n - index = 4 - 2 = 2 < 7`（`b` 超出 `a`）
- 需要至少 `2` 次 `a`，再額外加 `1` 次以完全覆蓋。

**結果**：`3`

---

### **範例 2**

python

複製編輯

`a = "abc" b = "cabcabca"`

#### **Step 1: 滾動哈希搜尋 `b`**

- `b = "cabcabca"` 的哈希值：計算 `hash_needle`
- `a + a + a = "abcabcabc"`，檢查 `b` 是否存在

|`i`|`hash_haystack`|是否匹配 `hash_needle`|
|---|---|---|
|0|不匹配|❌|
|1|不匹配|❌|
|2|匹配 `"cabcabca"`|✅|

- `index = 2`

#### **Step 2: 計算 `a` 需要重複的次數**

- `len(b) = 8`，`len(a) = 3`
- `n - index = 3 - 2 = 1 < 8`
- 至少 `3` 次 `a` 覆蓋 `b`。

**結果**：`3`

---

## **時間與空間複雜度分析**

- **`strstr()` 使用 Rabin-Karp 滾動哈希**
    
    - **計算 `hash_needle` 和 `hash_haystack`：`O(m)`**
    - **滑動哈希搜索：`O(n + m)`**
    - **總體 `O(n + m)`**
- **`repeated_string_match()`**
    
    - `strstr(a, b)` 的時間複雜度為 `O(n + m)`
    - 計算重複次數為 `O(1)`
    - **總體 `O(n + m)`**
- **空間複雜度**
    
    - `O(1)`，只使用了常數變數來計算哈希

---

## **其他解法想法（不含代碼）**

4. **暴力法 (`O(n * m)`)**
    
    - 直接在 `a` 重複多次後的字串中搜尋 `b`
    - 時間複雜度高，對大 `n, m` 不適用
5. **KMP (`O(n + m)`)**
    
    - 計算 `b` 的 KMP 前綴函數，然後用 KMP 在 `a` 重複後的字串中搜尋 `b`
    - 與 Rabin-Karp 方法類似，但在某些情況下更快
6. **使用 `b in (a * k)` 判斷 (`O(n + m)`)**
    
    - 直接構造 `a * k`，然後用 `"b in (a * k)"` 判斷 `b` 是否存在
    - 簡單但記憶體消耗較大

---

## **結論**

- **最佳解法**：Rabin-Karp (`O(n + m)`)
    - **利用滾動哈希高效匹配 `b`**
    - **只需遍歷 `a` 和 `b` 一次**
- **其他方法**
    - **KMP (`O(n + m)`)**：可行但較複雜
    - **暴力搜尋 (`O(n * m)`)**：不可行
    - **直接構造 `(a * k)` 再判斷 (`O(n + m)`)**：簡單但占用較多記憶體

本解法 **能夠高效找到 `b` 在 `a` 重複後的最小次數，並且避免暴力搜尋的效能問題**。