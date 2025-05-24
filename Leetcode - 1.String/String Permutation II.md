Lintcode 10
给出一个字符串，找到它的所有排列，注意同一个字符串不要打印两次。

**样例 1：**
输入：
```python
"""
s = "abb"
```
输出：
```python
"""
["abb", "bab", "bba"]
```
解释：
abb的全排列有6种，其中去重后还有3种。

**样例 2：**
输入：
```python
"""
s = "aabb"
```
输出：
```python
"""
["aabb", "abab", "baba", "bbaa", "abba", "baab"]
```


```python
    def string_permutation2(self, str):
        chars = sorted(list(str))
        visited = [False] * len(chars)
        permutations = []
        self.dfs(chars, visited, [], permutations) 
        return permutations

    def dfs(self, chars, visited, permutation, permutations):
        if len(chars) == len(permutation):
            permutations.append(''.join(permutation))
            return    
        
        for i in range(len(chars)):
            if visited[i]:
                continue

            if i > 0 and chars[i] == chars[i - 1] and not visited[i - 1]:
                continue

            visited[i] = True
            permutation.append(chars[i])

            self.dfs(chars, visited, permutation, permutations)

            permutation.pop()
            visited[i] = False
```
pass


本題 **`string_permutation2(self, str) -> List[str]`** 目標是找出 `str` **所有不重複的排列組合 (permutations)**，並以字典序排序返回。

### **核心觀察**

1. **全排列的基礎**
    
    - `n` 個字符的排列總共有 `n!` 種可能。
    - **若有重複字符，則需去重**，否則會產生**相同的排列**。
2. **避免重複排列**
    
    - 使用 **`visited` 陣列** 避免重複使用相同的字符。
    - **若前一個相同字符未被選擇 (`visited[i - 1] == False`)，則跳過當前字符**，避免重複。
3. **字典序排列**
    
    - 先對 `chars` **排序**，確保遞歸時的選擇是 **按字典序進行**。
4. **回溯 (Backtracking)**
    
    - 每次選擇一個字符加入 `permutation`，遞歸進入下一層。
    - 回溯時**恢復狀態** (`visited[i] = False`，並移除 `permutation` 的最後一個字符)。

---

## **變數表**

|變數名稱|含義|
|---|---|
|`chars`|`str` 轉換後的**排序列表** (確保字典序)|
|`visited`|記錄每個字符是否已被選擇 (避免重複使用)|
|`permutation`|當前排列的部分結果 (遞歸生成)|
|`permutations`|儲存所有完整的排列結果|

---

## **具體步驟**

1. **將 `str` 轉換為 `chars` 並排序**
    
    - 確保遞歸時的選擇是 **字典序**。
2. **使用回溯 (DFS) 生成排列**
    
    - **若 `permutation` 長度與 `chars` 相同，則將 `''.join(permutation)` 加入 `permutations`**。
3. **遍歷 `chars` 選擇下一個字符**
    
    - **若該字符已被使用 (`visited[i] == True`)，則跳過**。
    - **去重處理**：當 `chars[i] == chars[i - 1]`，且 `chars[i - 1]` 未被選擇 (`visited[i - 1] == False`)，則跳過。
4. **回溯 (Backtracking)**
    
    - 遞歸返回後，恢復狀態 (`permutation.pop()` & `visited[i] = False`)。

---

## **範例解析**

### **範例 1**

`str = "aab"`
#### **Step 1: 初始變數**

```python
"""
chars = ['a', 'a', 'b']
visited = [False, False, False]
permutations = []
```
#### **遞歸過程**

|遞歸深度|`permutation`|`visited` 狀態|`permutations`|
|---|---|---|---|
|0|`[]`|`[F, F, F]`|`[]`|
|1|`['a']`|`[T, F, F]`|`[]`|
|2|`['a', 'a']`|`[T, T, F]`|`[]`|
|3|`['a', 'a', 'b']`|`[T, T, T]`|`['aab']`|
|**回溯**|`['a', 'b']`|`[T, F, T]`|`['aab']`|
|3|`['a', 'b', 'a']`|`[T, T, T]`|`['aab', 'aba']`|
|**回溯**|`['b']`|`[F, F, T]`|`['aab', 'aba']`|
|2|`['b', 'a']`|`[T, F, T]`|`['aab', 'aba']`|
|3|`['b', 'a', 'a']`|`[T, T, T]`|`['aab', 'aba', 'baa']`|

**結果**：`["aab", "aba", "baa"]`

---

### **範例 2**

python

複製編輯

`str = "aba"`

|遞歸深度|`permutation`|`visited` 狀態|`permutations`|
|---|---|---|---|
|0|`[]`|`[F, F, F]`|`[]`|
|1|`['a']`|`[T, F, F]`|`[]`|
|2|`['a', 'b']`|`[T, T, F]`|`[]`|
|3|`['a', 'b', 'a']`|`[T, T, T]`|`['aba']`|
|**回溯**|`['b']`|`[F, T, F]`|`['aba']`|
|2|`['b', 'a']`|`[T, T, F]`|`['aba']`|
|3|`['b', 'a', 'a']`|`[T, T, T]`|`['aba', 'baa']`|

**結果**：`["aba", "baa"]`

---

## **時間與空間複雜度分析**

- **時間複雜度**
    
    - 遞歸生成所有排列：**`O(n!)`**
    - 排序 `O(n log n)`
    - **總體 `O(n! * n)`**（但通常 `O(n!)` 主導）
- **空間複雜度**
    
    - `visited` 陣列：`O(n)`
    - `permutation` 遞歸棧：最深 `O(n)`
    - `permutations` 儲存 `O(n!)` 個排列
    - **總體 `O(n!)`**

---

## **其他解法想法（不含代碼）**

1. **內建函數 `itertools.permutations` (`O(n!)`)**
    
    - 使用 `itertools.permutations(str)` 直接生成所有排列，再用 `set()` 去重。
2. **暴力生成所有排列 + 哈希去重 (`O(n! * n)`)**
    
    - 遞歸生成所有 `n!` 個排列，然後用 `set()` 過濾重複。
3. **字典樹 + 回溯 (`O(n!)`)**
    
    - 使用字典樹 (Trie) 存儲生成的排列，避免重複排列插入 `permutations`。

---

## **結論**

- **最佳方法**：**回溯 + `visited` 陣列去重 (`O(n!)`)**
- **可替代方法**
    - **`itertools.permutations` (`O(n!)`)**：最簡單但需要額外 `set()` 去重。
    - **暴力 `set()` 去重 (`O(n! * n)`)**：較慢但直觀。
    - **字典樹 (`O(n!)`)**：適合更大規模的數據。

**回溯法 (Backtracking) 是本題的最佳解法，透過 `visited` 陣列和去重條件，高效生成所有不重複的排列。** 🚀



```python
    def string_permutation2(self, str):
        chars = sorted(list(str))
        visited = [False] * len(chars)
        permutations = []
        self.dfs(chars, visited, [], permutations) 
        return permutations

    # 递归的定义: 找到所有 permutation 开头的排列
    def dfs(self, chars, visited, permutation, permutations):
        # 递归的出口：当我找到一个完整的排列
        if len(chars) == len(permutation):
            permutations.append(''.join(permutation))
            return    
        
        # 递归的拆解：基于当前的前缀，下一个字符放啥
        for i in range(len(chars)):
            # 同一个位置上的字符用过不能在用
            if visited[i]:
                continue
            # 去重：不同位置的同样的字符，必须按照顺序用。
            # a' a" b
            # => a' a" b => √
            # => a" a' b => x
            # 不能跳过一个a选下一个a
            if i > 0 and chars[i] == chars[i - 1] and not visited[i - 1]:
                continue

            # make changes
            visited[i] = True
            permutation.append(chars[i])

            # 找到所有 permutation 开头的排列
            # 找到所有 "a" 开头的
            self.dfs(chars, visited, permutation, permutations)

            # backtracking
            permutation.pop()
            visited[i] = False
```