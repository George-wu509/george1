Lintcode 1280
给定一个非负整数的数据流输入 a1，a2，…，an，…，将到目前为止看到的数字总结为不相交的区间列表。

例如，假设数据流中的整数为 1，3，7，2，6，…，每次的总结为：

```python
"""
[1, 1]
[1, 1], [3, 3]
[1, 1], [3, 3], [7, 7]
[1, 3], [7, 7]
[1, 3], [6, 7]
```
解釋:
當 數據流裡面加入1, 3, 7  可以看成區間 [1,1], [3,3], [7,7]
當加入2時, 因為和1,3相鄰就合併為[1,3]
所以結果就是[1,3], [7,7]

**Example 1:**
```python
"""
输入：
addNum(1)
getIntervals()
addNum(3)
getIntervals()
addNum(7)
getIntervals()
addNum(2)
getIntervals()
addNum(6)
getIntervals()
输出：
[[(1,1)],[(1,1),(3,3)],[(1,1),(3,3),(7,7)],[(1,3),(7,7)],[(1,3),(6,7)]]
解释：
addNum(1)
getIntervals([[1, 1]])
addNum(3)
getIntervals([[1, 1], [3, 3]])
addNum(7)
getIntervals([[1, 1], [3, 3], [7, 7]])
addNum(2)-merge(1,2,3)
getIntervals([[1, 3], [7, 7]])
addNum(6)->merge(6,7)
getIntervals([[1, 3], [6, 7]])
```
**Example 2:**
```python
"""
输入：
addNum(1)
getIntervals([[1, 1]])
addNum(3)
getIntervals([[1, 1], [3, 3]])
输出：
[null,null,[4,4],null,[3,4]]
解释：
addNum(4)
getIntervals([[4, 4]])
addNum(3)->merge(3,4)
getIntervals([[3, 4]])
```


```python
class Solution:
    
    def __init__(self):
        self.father = dict()
        self.val2interval = dict()

    def addNum(self, val):
        if val in self.father:
            return
        
        self.father[val] = val
        self.val2interval[val] = [val, val]
        if val - 1 in self.father:
            self.merge(val - 1, val)
            
        if val + 1 in self.father:
            self.merge(val + 1, val)
      
    def getIntervals(self):
        return [
            Interval(self.val2interval[val][0], self.val2interval[val][1])
            for val in sorted(self.father.keys())
            if self.father[val] == val
        ]
        
    def merge(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        self.father[root_a] = root_b
        self.val2interval[root_b] = [
            min(self.val2interval[root_a][0], self.val2interval[root_b][0]),
            max(self.val2interval[root_a][1], self.val2interval[root_b][1]),
        ]
    
    def find(self, a):
        if a == self.father[a]:
            return a
        self.father[a] = self.find(self.father[a])
        return self.father[a]
```
pass


# **LintCode 1280: Data Stream as Disjoint Intervals 解法分析**

## **解題目標**

我們要設計一個數據結構，能夠處理 **數據流的新增數字**，並且能夠 **維護不相交的區間 (disjoint intervals)**。

需要實作以下兩個函數：

1. **`addNum(val)`**
    
    - **插入 `val`** 到數據流中。
    - 需要合併與 `val` 相鄰的區間（若存在）。
    - **確保區間互不相交**。
2. **`getIntervals()`**
    
    - **返回目前數據流中的所有不相交區間**（按遞增順序排列）。

---

## **解法核心**

### **為何使用「並查集 (Union-Find) + 哈希表」？**

處理這類 **動態合併區間** 的問題時，傳統數據結構如 **二叉搜索樹 (`O(log n)`) 或有序字典 (`O(log n)`)** 雖然可行，但 **並查集 (Union-Find) 搭配哈希表可以更快地合併區間 (`O(α(n))`，接近 `O(1)`)**。

並查集 (`Union-Find`) 的主要優勢：

1. **快速查找某個數 `val` 屬於哪個區間 (`O(1)`)**。
2. **快速合併相鄰區間 (`O(1)`)**。
3. **動態維護區間範圍 (`min, max`)**，減少排序與遍歷的時間。

---

## **解法步驟**

### **Step 1: 初始化**

python

複製編輯

`def __init__(self):     self.father = dict()  # 並查集：每個數的父節點     self.val2interval = dict()  # 哈希表：記錄每個區間的 [start, end]`

- **`father` (`dict`)**：用來存儲 **每個數字對應的根節點**，即它所在的區間代表。
- **`val2interval` (`dict`)**：存儲 **區間的起點與終點**，即 `{root: [區間起點, 區間終點]}`。

---

### **Step 2: `addNum(val)`**

當插入 `val` 時：

1. 若 `val` **已經存在**，直接返回。
2. **初始化 `val` 為自己的父節點**（即它自己形成一個獨立的區間 `[val, val]`）。
3. 若 `val - 1` 或 `val + 1` 存在於 `father`，則與相鄰區間合併。

python

複製編輯

`def addNum(self, val):     if val in self.father:         return      self.father[val] = val     self.val2interval[val] = [val, val]      if val - 1 in self.father:         self.merge(val - 1, val)      if val + 1 in self.father:         self.merge(val + 1, val)`

---

### **Step 3: `merge(a, b)` (合併相鄰區間)**

當 `a, b` 相鄰時：

1. 找到 `a` 和 `b` **所屬的區間代表 (root)**。
2. **讓 `root_a` 指向 `root_b`**，即將 `a` 所屬區間合併到 `b` 所屬區間。
3. **更新 `root_b` 的區間範圍**，確保新的區間範圍為 `[min(left), max(right)]`。

python

複製編輯

`def merge(self, a, b):     root_a = self.find(a)     root_b = self.find(b)      self.father[root_a] = root_b  # 讓 root_a 指向 root_b      self.val2interval[root_b] = [         min(self.val2interval[root_a][0], self.val2interval[root_b][0]),         max(self.val2interval[root_a][1], self.val2interval[root_b][1]),     ]`

---

### **Step 4: `find(a)` (查找區間代表)**

**並查集 `find` 操作 (路徑壓縮)**：

- 如果 `a` 是自己的父節點，返回 `a`。
- **路徑壓縮**：讓 `a` 直接指向它的最終父節點，加快查找速度。

python

複製編輯

`def find(self, a):     if a == self.father[a]:         return a     self.father[a] = self.find(self.father[a])     return self.father[a]`

---

### **Step 5: `getIntervals()`**

1. 遍歷 `self.father.keys()`，確保只選擇區間代表 (`self.father[val] == val`)。
2. 按照數值順序排序，並返回區間 `[start, end]`。

python

複製編輯

`def getIntervals(self):     return [         Interval(self.val2interval[val][0], self.val2interval[val][1])         for val in sorted(self.father.keys()) if self.father[val] == val     ]`

---

## **具體範例**

### **範例 1**

text

複製編輯

`操作: addNum(1), addNum(3), addNum(7), getIntervals()`

**Step 1: 插入 `1`**

text

複製編輯

`Intervals: [ [1,1] ]`

**Step 2: 插入 `3`**

text

複製編輯

`Intervals: [ [1,1], [3,3] ]`

**Step 3: 插入 `7`**

text

複製編輯

`Intervals: [ [1,1], [3,3], [7,7] ]`

**Step 4: `getIntervals()`**

text

複製編輯

`返回 [ [1,1], [3,3], [7,7] ]`

---

### **範例 2**

text

複製編輯

`操作: addNum(1), addNum(2), getIntervals(), addNum(3), getIntervals()`

**Step 1: 插入 `1`**

text

複製編輯

`Intervals: [ [1,1] ]`

**Step 2: 插入 `2`（合併 `1` 和 `2`）**

text

複製編輯

`Intervals: [ [1,2] ]`

**Step 3: `getIntervals()`**

text

複製編輯

`返回 [ [1,2] ]`

**Step 4: 插入 `3`（合併 `1-2` 和 `3`）**

text

複製編輯

`Intervals: [ [1,3] ]`

**Step 5: `getIntervals()`**

text

複製編輯

`返回 [ [1,3] ]`

---

## **時間與空間複雜度分析**

### **時間複雜度**

|操作|時間複雜度|說明|
|---|---|---|
|`addNum(val)`|`O(α(n)) ≈ O(1)`|並查集 `find/union` 最優化 (`O(1)`)|
|`merge(a, b)`|`O(1)`|並查集 `find` + `union`|
|`getIntervals()`|`O(n log n)`|需要排序區間|

---

### **空間複雜度**

- **並查集 (`self.father`)** 需要 `O(n)`。
- **區間哈希表 (`self.val2interval`)** 需要 `O(n)`。
- **總計：`O(n)`**。

---

## **其他解法 (不寫 Code)**

1. **`OrderedDict` (`O(n log n)`)**
    
    - 用 **有序字典 (SortedDict)** 來維護區間。
    - **查詢與合併 `O(log n)`**，較慢。
2. **`Balanced BST` (紅黑樹) (`O(log n)`)**
    
    - **插入 + 合併 `O(log n)`**，適用於數據量較小的情境。
3. **暴力遍歷 (`O(n^2)`)**
    
    - 每次插入 `O(n)` 遍歷並合併，時間過長。

---

## **總結**

|**解法**|**時間複雜度**|**適用場景**|**優缺點**|
|---|---|---|---|
|**並查集 (Union-Find) (`O(α(n)) ≈ O(1)`)**|`O(1)`|**大規模數據流**|✅ 高效合併，動態維護區間|
|**OrderedDict (`O(n log n)`)**|`O(log n)`|小範圍數據|⚠ 排序成本較高|
|**BST (`O(log n)`)**|`O(log n)`|需要範圍查詢|⚠ 編碼較複雜|
|**暴力遍歷 (`O(n^2)`)**|`O(n^2)`|數據量小|❌ 低效|

✅ **最佳選擇：並查集 (`O(1)`)，適合所有場景！** 🚀