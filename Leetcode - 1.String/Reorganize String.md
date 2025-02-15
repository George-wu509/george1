Lintcode 1041
给定一个字符串`S`，检查是否能重新排布其中的字母，使得两相邻的字符不同。

若可行，输出任意可行的结果。若不可行，返回空字符串。

**样例 1:**
```python
"""
输入: S = "aab"
输出: "aba"
```
**样例 2:**
```python
"""
输入: S = "aaab"
输出: ""
```


```python
class Solution:
    def reorganize_string(self, s: str) -> str:
        if len(s) < 2:
            return s

        length = len(s)
        counts = collections.Counter(s)
        maxCount = max(counts.items(), key=lambda x: x[1])[1]
        if maxCount > (length + 1) // 2:
            return ""

        reorganizeArray = [""] * length
        evenIndex, oddIndex = 0, 1
        halfLength = length // 2

        for c, count in counts.items():
            while count > 0 and count <= halfLength and oddIndex < length:
                reorganizeArray[oddIndex] = c
                count -= 1
                oddIndex += 2
            while count > 0:
                reorganizeArray[evenIndex] = c
                count -= 1
                evenIndex += 2

        return "".join(reorganizeArray)
```
pass

## **解法思路**

本題 **`reorganize_string(self, s: str) -> str`** 的目標是：

- **重新排列 `s`，使得相鄰的字母不相同**。
- **若無法達成，則返回空字串 `""`**。

---

## **解法分析**

### **核心想法**

1. **統計字母出現頻率 (`collections.Counter`)**
    
    - 若某個字母的頻率 `maxCount` 超過 `(length + 1) // 2`，則無法避免相鄰重複，直接返回 `""`。
2. **優先安排最多的字母**
    
    - **優先將最多的字母填入偶數索引 (`0, 2, 4, ...`)**。
    - 若偶數索引填滿，則填入奇數索引 (`1, 3, 5, ...`)。
3. **確保不相鄰**
    
    - **由於最多的字母先填入偶數索引，因此後續字母自動填入剩餘位置，確保不會有連續相同的字母**。

---

## **變數表**

|變數名稱|含義|
|---|---|
|`s`|原始字串|
|`length`|`s` 的長度|
|`counts`|統計 `s` 中每個字母的頻率 (`Counter`)|
|`maxCount`|`s` 中出現次數最多的字母的頻率|
|`reorganizeArray`|存放最終重新排列的字串|
|`evenIndex`|當前可填入偶數索引 (`0, 2, 4, ...`)|
|`oddIndex`|當前可填入奇數索引 (`1, 3, 5, ...`)|
|`halfLength`|`length // 2`，用來控制字母放置的上限|

---

## **具體步驟**

### **Step 1: 檢查是否可重新排列**

- **使用 `Counter` 計算 `s` 中每個字母的頻率**。
- **找到 `maxCount` (出現最多次的字母的頻率)**。
- 若 `maxCount > (length + 1) // 2`，則無法避免相鄰相同，返回 `""`。

### **Step 2: 重新排列**

4. **初始化 `reorganizeArray`，大小為 `length`，用來存儲結果**。
5. **優先填充最多的字母**
    - 先填入 `偶數索引` (`0, 2, 4, ...`)，避免相鄰相同。
    - **當偶數索引填滿後，才開始填入奇數索引 (`1, 3, 5, ...`)**。
6. **遍歷 `counts.items()`，按字母頻率將其依序填入 `reorganizeArray`**。

### **Step 3: 返回結果**

- 最終將 `reorganizeArray` 轉成字串，返回 `''.join(reorganizeArray)`。

---

## **範例解析**

### **範例 1**

`s = "aab"`

**Step 1: 統計字母頻率**

`counts = {'a': 2, 'b': 1} maxCount = 2 length = 3`

`maxCount (2) ≤ (3 + 1) // 2 = 2`，可以重新排列。

**Step 2: 重新排列**

|填充順序|`reorganizeArray`|`evenIndex`|`oddIndex`|
|---|---|---|---|
|放 `a`|`["a", "", ""]`|2|1|
|放 `a`|`["a", "", "a"]`|4|1|
|放 `b`|`["a", "b", "a"]`|4|3|

**結果**：

python

複製編輯

`"a b a"`

**返回 `"aba"`**。

---

### **範例 2**

python

複製編輯

`s = "aaab"`

**Step 1: 統計字母頻率**

python

複製編輯

`counts = {'a': 3, 'b': 1} maxCount = 3 length = 4`

**檢查是否可重新排列**

python

複製編輯

`maxCount (3) > (4 + 1) // 2 = 2`

**無法避免相鄰 `"aa"`，返回 `""`**。

---

### **範例 3**

python

複製編輯

`s = "aabb"`

**Step 1: 統計字母頻率**

python

複製編輯

`counts = {'a': 2, 'b': 2} maxCount = 2 length = 4`

`maxCount (2) ≤ (4 + 1) // 2 = 2`，可以重新排列。

**Step 2: 重新排列**

|填充順序|`reorganizeArray`|`evenIndex`|`oddIndex`|
|---|---|---|---|
|放 `a`|`["a", "", "", ""]`|2|1|
|放 `a`|`["a", "", "a", ""]`|4|1|
|放 `b`|`["a", "b", "a", ""]`|4|3|
|放 `b`|`["a", "b", "a", "b"]`|4|5|

**結果**：

python

複製編輯

`"a b a b"`

**返回 `"abab"`**。

---

## **時間與空間複雜度分析**

- **時間複雜度**
    
    1. **計算頻率 (`Counter(s)`)：`O(N)`**
    2. **找到 `maxCount`：`O(26) ≈ O(1)`** (因為最多 26 種字母)
    3. **遍歷 `counts.items()` 來填充結果：`O(N)`**
    4. **總體時間複雜度：`O(N)`**
- **空間複雜度**
    
    1. **`Counter` 佔用 `O(26) ≈ O(1)`**
    2. **`reorganizeArray` 佔用 `O(N)`**
    3. **總體空間複雜度：`O(N)`**

---

## **其他解法想法（不含代碼）**

7. **使用 `heapq`（最大堆 `O(N log N)`)**
    
    - 透過 **最大堆 (Max-Heap)** 來存放頻率最高的字母。
    - 逐步取出最常見的字母，並插入間隔的位置，確保不相鄰。
8. **貪心法 + 排序 (`O(N log N)`)**
    
    - 先對 `counts` 按頻率排序 (`O(N log N)`)。
    - 按照降序插入結果字串，確保最常見的字母有最大間隔。
9. **暴力回溯 (`O(N!)`)**
    
    - 透過遞歸嘗試所有可能的排列，找到符合條件的結果。
    - **時間複雜度極高，完全不可行！**

---

## **結論**

- **最佳方法**：**貪心法 + 偶數索引優先 (`O(N)`)**
    - **時間 `O(N)`，空間 `O(N)`**
    - **高效處理，適合大數據集**
- **可替代方法**
    - **最大堆 (`O(N log N)`)**：適用於字母種類較多的情況。
    - **貪心排序 (`O(N log N)`)**：易於理解，但稍慢。
    - **暴力回溯 (`O(N!)`)**：不可行。

🚀 **此方法利用 `Counter` 計數 + 優先填充偶數索引，達到 `O(N)` 高效解法，適用於大數據集！** 🚀

  

O