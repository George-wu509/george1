Lintcode 1740
编写一个 `StockSpanner` 类，它收集某些股票的每日报价，并返回该股票当日价格的跨度。

今天股票价格的跨度被定义为股票价格小于或等于今天价格的最大连续日数（从今天开始往回数，包括今天）。

例如，如果未来7天股票的价格是 `[100, 80, 60, 70, 60, 75, 85]`，那么股票跨度将是 `[1, 1, 1, 2, 1, 4, 6]`。


**样例 1:**
```python
"""
输入：prices = [100,80,60,70,60,75,85]
输出：[1,1,1,2,1,4,6]
解释：
首先，初始化 S = StockSpanner()，然后：
S.next(100) 被调用并返回 1，
S.next(80) 被调用并返回 1，
S.next(60) 被调用并返回 1，
S.next(70) 被调用并返回 2，
S.next(60) 被调用并返回 1，
S.next(75) 被调用并返回 4，
S.next(85) 被调用并返回 6。

注意 (例如) S.next(75) 返回 4，因为截至今天的最后 4 个价格
(包括今天的价格 75) 小于或等于今天的价格。
```
**样例 2:**

```python
"""
输入：prices = [50,80,80,70,90,75,85]
输出：[1,2,3,1,5,1,2]
解释：
首先，初始化 S = StockSpanner()，然后：
S.next(50) 被调用并返回 1，
S.next(80) 被调用并返回 2，
S.next(80) 被调用并返回 3，
S.next(70) 被调用并返回 1，
S.next(90) 被调用并返回 5，
S.next(75) 被调用并返回 1，
S.next(85) 被调用并返回 2。
```


```python
class StockSpanner(object):
    def __init__(self):
        self.stack = []

    def next(self, price):
        weight = 1
        while self.stack and self.stack[-1][0] <= price:
            weight += self.stack.pop()[1]
        self.stack.append((price, weight))
        return weight
```
pass


### **LintCode 1740 - Online Stock Span**

#### **解法分析**

本題的目標是設計一個 `StockSpanner` 類別，它支援 `next(price)` 方法，該方法會返回**當前股價 `price` 的連續遞增天數**（包含當天），也就是**向左看能夠持續小於等於 `price` 的天數總和**。

例如：

`輸入: [100, 80, 60, 70, 60, 75, 85] 輸出: [1, 1, 1, 2, 1, 4, 6]`

這是因為：

- `100` 本身就大 → `1`
- `80` 沒有更小的過去股價 → `1`
- `60` 沒有更小的過去股價 → `1`
- `70` 可以包括自己和 `60` → `2`
- `60` 沒有更小的過去股價 → `1`
- `75` 可以包含自己、`70`、`60` → `4`
- `85` 可以包含自己、`75`、`70`、`60`、`80` → `6`

---

### **解法思路**

**使用單調遞減棧 (Monotonic Decreasing Stack)** 來加速查詢過去比 `price` 小的天數：

1. **使用棧 `stack`** 存儲 `(price, weight)`：
    
    - `price` 是股價
    - `weight` 是 `price` 作為當前最大值時，向左可以延伸的天數（包含自己）
2. **計算 `next(price)`**
    
    - 設定 `weight = 1`（包含當前天）
    - 若 `stack` 不為空且 `stack[-1][0] <= price`，則：
        - 代表棧頂元素的股價 **小於等於當前股價**，可以合併這段區間
        - 累加 `weight += stack.pop()[1]`
    - **壓入新 `(price, weight)`**
    - 返回 `weight`

---

### **變數說明**

|變數名稱|說明|
|---|---|
|`stack`|單調遞減棧，存儲 `(price, weight)`|
|`price`|當前輸入的股價|
|`weight`|當前股價能影響的天數|

---

### **範例**

#### **輸入**

python

複製編輯

`stockSpanner = StockSpanner() print(stockSpanner.next(100)) # 1 print(stockSpanner.next(80))  # 1 print(stockSpanner.next(60))  # 1 print(stockSpanner.next(70))  # 2 print(stockSpanner.next(60))  # 1 print(stockSpanner.next(75))  # 4 print(stockSpanner.next(85))  # 6`

#### **處理流程**

1. `next(100)`
    - `stack = [(100, 1)]`
    - `return 1`
2. `next(80)`
    - `stack = [(100, 1), (80, 1)]`
    - `return 1`
3. `next(60)`
    - `stack = [(100, 1), (80, 1), (60, 1)]`
    - `return 1`
4. `next(70)`
    - `60` 小於 `70`，合併 `weight = 2`
    - `stack = [(100, 1), (80, 1), (70, 2)]`
    - `return 2`
5. `next(60)`
    - `stack = [(100, 1), (80, 1), (70, 2), (60, 1)]`
    - `return 1`
6. `next(75)`
    - `60`、`70` 小於 `75`，合併 `weight = 4`
    - `stack = [(100, 1), (80, 1), (75, 4)]`
    - `return 4`
7. `next(85)`
    - `75`、`80` 小於 `85`，合併 `weight = 6`
    - `stack = [(100, 1), (85, 6)]`
    - `return 6`

#### **輸出**

python

複製編輯

`1, 1, 1, 2, 1, 4, 6`

---

### **時間與空間複雜度分析**

#### **時間複雜度**

- 每個 `price` **最多進棧一次，出棧一次**，所以均攤時間複雜度為 **O(1)**。
- 若是 `N` 個 `price`，總時間為 **O(N)**。

#### **空間複雜度**

- `stack` 最壞情況下存儲 `N` 個元素，為 **O(N)**。

---

### **其他解法想法**

8. **暴力解法 (O(N²))**
    
    - 每次 `next(price)` 向左遍歷 **找到最大連續遞增數量**。
    - **時間複雜度 O(N²)**，不適合大量數據。
9. **使用隊列 (O(N))**
    
    - 若數據是離線批量處理，可使用 **雙端隊列 (Deque)** 來加速查詢。
    - **時間複雜度 O(N)**，但在線模式不適合。
10. **動態窗口 (O(N))**
    
    - 若股價變動幅度受限，可利用 **動態滑動窗口** 來快速查詢。
    - **時間複雜度 O(N)**。

---

### **總結**

- **最優解法：單調棧 (O(1) 均攤時間, O(N) 空間)**
- **若 `N` 小，可使用暴力解法 (O(N²))**
- **若需要批量處理，可考慮 `Deque` 或 `Sliding Window` (O(N))**