Lintcode 1435
对于某些字符串 `S`，我们将执行一些替换操作，用新的字母组替换原有的字母组（不一定大小相同）。

每个替换操作具有 3 个参数：起始索引 `i`，源字 `x` 和目标字 `y`。规则是如果 `x` 从**原始字符串 S** 中的位置 `i` 开始，那么我们将用 `y` 替换出现的 `x`。如果没有，我们什么都不做。

举个例子，如果我们有 `S = “abcd”` 并且我们有一些替换操作 `i = 2，x = “cd”，y = “ffff”`，那么因为 `“cd”` 从原始字符串 `S` 中的位置 `2` 开始，我们将用 `“ffff”` 替换它。

再来看 `S = “abcd”` 上的另一个例子，如果我们有替换操作 `i = 0，x = “ab”，y = “eee”`，以及另一个替换操作 `i = 2，x = “ec”，y = “ffff”`，那么第二个操作将不执行任何操作，因为原始字符串中 `S[2] = 'c'`，与 `x[0] = 'e'` 不匹配。

所有这些操作同时发生。保证在替换时不会有任何重叠： `S = "abc", indexes = [0, 1], sources = ["ab","bc"]` 不是有效的测试用例。


```python
"""
输入：S = "abcd", indexes = [0,2], sources = ["a","cd"], targets = ["eee","ffff"]
输出："eeebffff"
解释：
"a" 从 S 中的索引 0 开始，所以它被替换为 "eee"。
"cd" 从 S 中的索引 2 开始，所以它被替换为 "ffff"。
```
**样例 2:**
```python
"""
输入：S = "abcd", indexes = [0,2], sources = ["ab","ec"], targets = ["eee","ffff"]
输出："eeecd"
解释：
"ab" 从 S 中的索引 0 开始，所以它被替换为 "eee"。
"ec" 没有从原始的 S 中的索引 2 开始，所以它没有被替换。
```


```python
    def find_replace_string(self, s: str, indexes: List[int], sources: List[str], targets: List[str]) -> str:
        cur_i = 0
        result = []

        zipped = sorted(zip(indexes, sources, targets))

        for i in range(len(zipped)):
            index = zipped[i][0]
            source = zipped[i][1]
            target = zipped[i][2]

            if cur_i < index:
                result.append(s[cur_i:index])
                cur_i = index
            
            if s[index:index + len(source)] == source:
                result.append(target)
                cur_i += len(source)

        if cur_i < len(s):
            result.append(s[cur_i:])

        return ''.join(result)
```
pass


本題 **`find_replace_string(self, s: str, indexes: List[int], sources: List[str], targets: List[str]) -> str`** 的目標是：

- **在 `s` 中找到 `sources[i]`，並替換為 `targets[i]`，替換操作發生在 `indexes[i]`**。
- **若 `s[indexes[i]:indexes[i] + len(sources[i])] != sources[i]`，則跳過該替換**。

---

## **解法分析**

### **核心想法**

1. **將 `indexes, sources, targets` 按 `indexes` 升序排序**
    
    - 因為輸入 `indexes` 可能是無序的，排序確保替換順序正確。
2. **遍歷 `indexes`，檢查 `s` 的對應位置是否與 `sources` 匹配**
    
    - 若匹配，則用 `targets` 替換 `sources`。
3. **使用 `cur_i` 追蹤 `s` 當前處理的索引**
    
    - `cur_i` 代表當前處理到 `s` 的哪個位置，確保所有未替換的部分都加入結果。
4. **使用 `result` 來存儲最終結果**
    
    - 逐步拼接 `s` 的未修改部分和 `targets`，最後合併為字串。

---

## **變數表**

|變數名稱|含義|
|---|---|
|`s`|原始字串|
|`indexes`|需要替換的起始索引列表|
|`sources`|每個 `index` 需要被匹配的字串|
|`targets`|用來替換 `sources[i]` 的字串|
|`zipped`|`(index, source, target)` 按 `index` 排序的三元組|
|`cur_i`|當前處理到 `s` 的索引|
|`result`|存儲最終結果的陣列|

---

## **具體步驟**

### **Step 1: 對 `indexes` 進行排序**

- 因為 `indexes` 可能是無序的，我們需要確保 **從左到右順序替換**。
- **排序 `zipped = sorted(zip(indexes, sources, targets))`**，這樣可以確保 **按照 `indexes` 的順序替換**。

### **Step 2: 遍歷 `indexes`，執行替換**

- **對於每個 `(index, source, target)`**：
    1. **若 `cur_i < index`**：表示 `cur_i` 之後有未處理的 `s` 片段，先加入 `result`。
    2. **檢查 `s[index:index + len(source)] == source`**：
        - 若相等，則替換為 `target`，並更新 `cur_i` 到 `index + len(source)`。
        - 否則，跳過該替換。

### **Step 3: 處理剩餘的 `s`**

- 若 `cur_i < len(s)`，則將 `s[cur_i:]` 加入 `result`。

### **Step 4: 返回結果**

- `''.join(result)` 拼接 `result` 陣列，返回最終字串。

---

## **範例解析**

### **範例 1**

python

複製編輯

`s = "abcd" indexes = [0, 2] sources = ["a", "cd"] targets = ["eee", "ffff"]`

#### **Step 1: `zipped` 排序**

python

複製編輯

`zipped = [(0, "a", "eee"), (2, "cd", "ffff")]`

#### **Step 2: 替換過程**

|`i`|`index`|`source`|`target`|`s[index:index+len(source)]`|匹配？|`result` 更新|`cur_i`|
|---|---|---|---|---|---|---|---|
|0|0|"a"|"eee"|`"a"`|✅|`["eee"]`|1|
|1|2|"cd"|"ffff"|`"cd"`|✅|`["eee", "b", "ffff"]`|4|

#### **Step 3: 處理剩餘 `s`**

- `cur_i = 4`，已處理完 `s`，無需額外添加。

**結果**：

python

複製編輯

`"eeebffff"`

---

### **範例 2**

python

複製編輯

`s = "abcd" indexes = [0, 2] sources = ["ab", "ec"] targets = ["eee", "ffff"]`

#### **Step 1: `zipped` 排序**

python

複製編輯

`zipped = [(0, "ab", "eee"), (2, "ec", "ffff")]`

#### **Step 2: 替換過程**

|`i`|`index`|`source`|`target`|`s[index:index+len(source)]`|匹配？|`result` 更新|`cur_i`|
|---|---|---|---|---|---|---|---|
|0|0|"ab"|"eee"|`"ab"`|✅|`["eee"]`|2|
|1|2|"ec"|"ffff"|`"cd"`|❌|**跳過**|2|

#### **Step 3: 處理剩餘 `s`**

- `cur_i = 2`，添加 `"cd"`。

**結果**：

python

複製編輯

`"eeecd"`

---

## **時間與空間複雜度分析**

- **時間複雜度**
    
    - `zip()` + `sorted()`：`O(K log K)`，其中 `K = len(indexes)`。
    - 遍歷 `s` 和 `indexes` 替換：`O(N + K)`。
    - **總體時間複雜度**： O(Klog⁡K+N+K)=O(Klog⁡K+N)O(K \log K + N + K) = O(K \log K + N)O(KlogK+N+K)=O(KlogK+N)
- **空間複雜度**
    
    - `result` 陣列：`O(N)`
    - `zipped` 陣列：`O(K)`
    - **總體 `O(N + K)`**。

---

## **其他解法想法（不含代碼）**

1. **暴力遍歷 `s` (`O(NK)`)**
    
    - 直接遍歷 `s`，對 `indexes` 檢查是否需要替換。
    - **時間複雜度 `O(NK)`**，效率較低。
2. **使用 `dict` 優化匹配 (`O(N + K)`)**
    
    - 預先構建 `index -> (source, target)` 映射 (`O(K)`)
    - 遍歷 `s` 時，直接查詢 `index` 是否可替換。
    - **時間 `O(N + K)`，比排序方式快**。
3. **直接構造 `s` 的替換表 (`O(N + K)`)**
    
    - 創建 `replace_map`，標記 `s` 需要替換的位置與對應 `target`。
    - 遍歷 `s` 時，根據 `replace_map` 替換或保留字符。

---

## **結論**

- **最佳方法**：**排序 `indexes` + `cur_i` 指針 (`O(K log K + N)`)**
    - **處理順序確保正確**。
    - **避免不必要的檢查，提高效率**。
- **可替代方法**
    - **暴力法 (`O(NK)`)**：時間較慢。
    - **哈希表 (`O(N + K)`)**：更快，但較難實作。
    - **構造替換表 (`O(N + K)`)**：適合超長 `s`。

**本解法是最通用、時間複雜度合理的方法，能高效處理 `s` 的替換問題**