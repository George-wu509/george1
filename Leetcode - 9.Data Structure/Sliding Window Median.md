
LintCode 360 的問題 **Sliding Window Median** 要求找到一個滑動窗口的中位數，當窗口從左至右滑動時，中位數應隨著窗口內容的變化而更新。這是一個經典的問題，涉及到數據結構的靈活應用。以下是問題的詳細解法、例子及複雜度分析。

## **LintCode 360：滑动窗口的中位数**

---

### **題目描述**

給定一個數組 `nums` 和一個整數 `k`，我們需要計算 **滑動窗口的中位數**。即：

- 每當窗口從 `nums[i:i+k]` 滑動時，求出當前窗口的中位數。

**示例**

`輸入： nums = [1,2,7,8,5], k = 3  輸出： [2, 7, 7]`

**解釋**

複製編輯

`窗口 [1,2,7] => 中位數 2 窗口 [2,7,8] => 中位數 7 窗口 [7,8,5] => 中位數 7`

---

## **解法：雙堆（最大堆 + 最小堆）**

### **核心思路**

1. **維護兩個堆**
    
    - **`max_heap`（最大堆）**：存儲窗口內 **較小的一半數字**，並將值取負來模擬最大堆（Python 默認是最小堆）。
    - **`min_heap`（最小堆）**：存儲窗口內 **較大的一半數字**。
2. **維護平衡**
    
    - 讓 `max_heap` 和 `min_heap` 的大小保持相近（相差不超過 1）。
    - 當 `min_heap` 元素比 `max_heap` 多時，將 `min_heap` 的最小值取出放到 `max_heap`。
3. **滑動窗口**
    
    - 插入當前數字到正確的堆（確保 `max_heap` 存較小的 `k/2` 個數，`min_heap` 存較大的 `k/2` 個數）。
    - 移除過期的數字（即 `nums[i-k]`）。
    - 調整堆的平衡，確保 `max_heap` 的大小 `≤ min_heap + 1`。
4. **取中位數**
    
    - 若 `max_heap` 比 `min_heap` 大，則 `max_heap` 的頂部就是中位數。
    - 若 `min_heap` 比 `max_heap` 大，則 `min_heap` 的頂部就是中位數。

---

### **代碼解析**

```python
from heapq import heappush, heappop

# 自定義堆類
class Heap:
    def __init__(self):
        self.heap = []
        self.deleted = {}  # 紀錄被刪除但尚未從堆中移除的元素
        self._len = 0

    def push(self, val):
        heappush(self.heap, val)
        self._len += 1

    def pop(self):
        self._clean_top()
        self._len -= 1
        return heappop(self.heap)

    def remove(self, val):
        self.deleted[val] = self.deleted.get(val, 0) + 1
        self._len -= 1  # 記錄刪除，但不立即從堆移除

    def top(self):
        self._clean_top()
        return self.heap[0]

    def _clean_top(self):
        while self.heap and self.deleted.get(self.heap[0]):
            self.deleted[self.heap[0]] -= 1
            heappop(self.heap)

    def __len__(self):
        return self._len

class Solution:
    def median_sliding_window(self, nums, k):
        ans = []
        if not nums or len(nums) < 1 or k <= 0:
            return ans

        self.min_heap = Heap()  # 存儲較大的數
        self.max_heap = Heap()  # 存儲較小的數（取負值）

        for i in range(len(nums)):
            # **移除過期元素**
            if i >= k:
                if len(self.min_heap) and nums[i - k] >= self.min_heap.top():
                    self.min_heap.remove(nums[i - k])
                else:
                    self.max_heap.remove(- nums[i - k])

            # **插入新元素**
            if len(self.min_heap) and nums[i] > self.min_heap.top():
                self.min_heap.push(nums[i])
            else:
                self.max_heap.push(- nums[i])  # 取負模擬最大堆

            self.balance()

            # **獲取中位數**
            if i + 1 >= k:
                ans.append(self.get_median())

        return ans

    # **維持平衡**
    def balance(self):
        l = len(self.max_heap)
        r = len(self.min_heap)
        if abs(r - l) <= 1:
            return
        if r > l:
            self.max_heap.push(- self.min_heap.pop())
        else:
            self.min_heap.push(- self.max_heap.pop())
        self.balance()

    # **獲取中位數**
    def get_median(self):
        l = len(self.max_heap)
        r = len(self.min_heap)
        if r > l:
            return self.min_heap.top()
        else:
            return - self.max_heap.top()

```
---

## **逐步執行分析**

### **輸入**

`nums = [1,2,7,8,5] k = 3`

### **初始化**

- `max_heap = []`（較小的一半數字，存負數）。
- `min_heap = []`（較大的一半數字）。

### **第一個窗口 [1,2,7]**

- 插入 `1`：放入 `max_heap → [-1]`
- 插入 `2`：放入 `min_heap → [2]`
- 插入 `7`：放入 `min_heap → [2, 7]`
- 平衡：移動 `2` 到 `max_heap`
- **中位數：2**

### **第二個窗口 [2,7,8]**

- 移除 `1`：從 `max_heap` 刪除
- 插入 `8`：放入 `min_heap → [7, 8]`
- **中位數：7**

### **第三個窗口 [7,8,5]**

- 移除 `2`：從 `max_heap` 刪除
- 插入 `5`：放入 `max_heap → [-5]`
- 平衡：移動 `7` 到 `max_heap`
- **中位數：7**

### **輸出**

`[2, 7, 7]`

---

## **時間與空間複雜度分析**

### **時間複雜度**

- 插入堆 `O(log k)`
- 刪除元素 `O(log k)`
- 獲取中位數 `O(1)`
- **總時間複雜度：`O(n log k)`**

### **空間複雜度**

- 兩個堆的大小為 `O(k)`
- **總空間複雜度：`O(k)`**

---

## **其他解法**

### **1. 暴力法（O(nk log k)）**

- 每次窗口變動時，重新排序並獲取中位數。
- 時間複雜度過高。

### **2. 使用平衡樹（O(n log k)）**

- 使用 `SortedList` 來維護窗口內的數字，獲取中位數為 `O(1)`。
```python
from sortedcontainers import SortedList

def median_sliding_window(nums, k):
    ans = []
    window = SortedList()

    for i in range(len(nums)):
        if i >= k:
            window.remove(nums[i - k])

        window.add(nums[i])

        if i + 1 >= k:
            ans.append(window[(k - 1) // 2])

    return ans

```
- **時間複雜度：`O(n log k)`**
- **適用場景：`k` 很小，`n` 很大時，比雙堆更高效。**

---

## **總結**

|方法|時間複雜度|空間複雜度|適用情境|
|---|---|---|---|
|**雙堆（最優解）**|`O(n log k)`|`O(k)`|**適用於一般情況**|
|暴力排序|`O(n k log k)`|`O(k)`|**小數據適用**|
|平衡樹|`O(n log k)`|`O(k)`|**適用於 `k` 小，`n` 大的情況**|

🚀 **雙堆法是這道題的最佳解法，平衡了查找與刪除的性能！**