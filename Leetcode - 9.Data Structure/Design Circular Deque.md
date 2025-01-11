
以下是LintCode 999題「用循环数组实现双向队列」的詳細解釋，包括具體的例子和每一步驟，並附上複雜度分析及其他解法。

---

### **題目分析**

設計一個支持以下操作的雙向隊列，並用循環數組實現：

1. **`isFull`**：檢查隊列是否已滿。
2. **`isEmpty`**：檢查隊列是否為空。
3. **`pushFront`**：在隊首插入元素。
4. **`popFront`**：從隊首刪除元素。
5. **`pushBack`**：在隊尾插入元素。
6. **`popBack`**：從隊尾刪除元素。

---

### **解法核心思路**

使用循環數組實現雙向隊列：

1. 定義數組 `stack` 作為底層存儲結構。
2. 使用兩個指針 `left` 和 `right` 分別表示隊首和隊尾位置。
3. 設置變量 `size` 來記錄當前隊列內的元素個數，方便檢查滿/空。

#### **關鍵點**

- **循環結構：**
    - 當指針到達數組邊界時，通過模運算回到數組的起始位置。
- **邊界條件：**
    - 插入或刪除操作需要考慮隊列是否滿/空。

Example:
```python
输入:
CircularDeque(5)
isFull()
isEmpty()
pushFront(1)
pushBack(2)
pushBack(3)
popFront()
popFront()
popBack()
输出:
["false","true","1","2","3"]
```
```python
输入:
CircularDeque(5)
pushFront(1)
pushFront(2)
pushFront(3)
popBack()
popBack()
popBack()
输出:
["3","2","1"]
```



```python
class CircularDeque:
    def __init__(self, n):
        # initialize your data structure here
        self.left = 0
        self.right = 0
        self.stack = [0] * n
        self.size = 0
    """
    @return:  return true if the array is full
    """
    def isFull(self):
        # write your code here
        return self.size == len(self.stack)

    """
    @return: return true if there is no element in the array
    """
    def isEmpty(self):
        # write your code here
        return self.size == 0

    """
    @param element: the element given to be added
    @return: nothing
    """
    def pushFront(self, element):
        # write your code here
        if self.isEmpty():
            self.left = self.right = 0
            self.stack[self.left] = element
        else:
            idx = self.left - 1
            if idx < 0:
                idx = len(self.stack) - 1
            self.left = idx
            self.stack[self.left] = element
        self.size += 1

    """
    @return: pop an element from the front of deque 
    """
    def popFront(self):
        # write your code here
        self.size -= 1
        val = self.stack[self.left]
        self.stack[self.left] = 0
        idx = self.left + 1
        if idx == len(self.stack):
            idx = 0
        self.left = idx
        return val

    """
    @param element: element: the element given to be added
    @return: nothing
    """
    def pushBack(self, element):
        # write your code here
        if self.isEmpty():
            self.left = self.right = 0
            self.stack[self.right] = element
        else:
            idx = self.right + 1
            if idx == len(self.stack):
                idx = 0
            self.right = idx
            self.stack[self.right] = element
        self.size += 1

    """
    @return: pop an element from the tail of deque
    """
    def popBack(self):
        # write your code here
        self.size -= 1
        val = self.stack[self.right]
        self.stack[self.right] = 0
        idx = self.right - 1
        if idx < 0:
            idx = len(self.stack) - 1
        self.right = idx
        return val
```
pass

---

### **解法步驟**

#### **1. 初始化**

- **參數：**
    - `n`：數組的最大容量。
- **初始化：**
    - `stack = [0] * n`：存儲數據的循環數組。
    - `left = 0`，`right = 0`：初始化指針。
    - `size = 0`：初始隊列大小為 0。

---

#### **2. 檢查操作**

- **`isFull`：**
    - 判斷條件：`size == len(stack)`。
- **`isEmpty`：**
    - 判斷條件：`size == 0`。

---

#### **3. 插入操作**

- **`pushFront`：**
    
    - 如果隊列為空，直接將元素插入隊首。
    - 如果不空：
        - 計算插入位置：`left = (left - 1 + len(stack)) % len(stack)`。
        - 更新 `stack[left]` 為新元素。
    - 增加 `size`。
- **`pushBack`：**
    
    - 如果隊列為空，直接將元素插入隊尾。
    - 如果不空：
        - 計算插入位置：`right = (right + 1) % len(stack)`。
        - 更新 `stack[right]` 為新元素。
    - 增加 `size`。

---

#### **4. 刪除操作**

- **`popFront`：**
    
    - 獲取隊首值：`val = stack[left]`。
    - 清空隊首位置：`stack[left] = 0`。
    - 更新隊首指針：`left = (left + 1) % len(stack)`。
    - 減少 `size`，返回 `val`。
- **`popBack`：**
    
    - 獲取隊尾值：`val = stack[right]`。
    - 清空隊尾位置：`stack[right] = 0`。
    - 更新隊尾指針：`right = (right - 1 + len(stack)) % len(stack)`。
    - 減少 `size`，返回 `val`。

---

### **具體舉例**

#### 初始化：

python

複製程式碼

`dq = CircularDeque(4)`

- `stack = [0, 0, 0, 0]`
- `left = 0`, `right = 0`, `size = 0`

#### 插入操作：

1. **`dq.pushFront(1)`：**
    
    - 插入 `1` 到隊首。
    - 更新：`stack = [1, 0, 0, 0]`，`left = 0`，`right = 0`，`size = 1`
2. **`dq.pushBack(2)`：**
    
    - 插入 `2` 到隊尾。
    - 更新：`stack = [1, 2, 0, 0]`，`left = 0`，`right = 1`，`size = 2`
3. **`dq.pushFront(3)`：**
    
    - 插入 `3` 到隊首。
    - 更新：`stack = [1, 2, 0, 3]`，`left = 3`，`right = 1`，`size = 3`
4. **`dq.pushBack(4)`：**
    
    - 插入 `4` 到隊尾。
    - 更新：`stack = [4, 2, 0, 3]`，`left = 3`，`right = 2`，`size = 4`

#### 刪除操作：

1. **`dq.popFront()`：**
    
    - 刪除隊首 `3`。
    - 更新：`stack = [4, 2, 0, 0]`，`left = 0`，`right = 2`，`size = 3`
    - 返回 `3`
2. **`dq.popBack()`：**
    
    - 刪除隊尾 `4`。
    - 更新：`stack = [0, 2, 0, 0]`，`left = 0`，`right = 1`，`size = 2`
    - 返回 `4`

---

### **時間與空間複雜度分析**

1. **時間複雜度：**
    
    - 插入與刪除操作的時間複雜度均為 O(1)O(1)O(1)，因為只涉及指針的計算和數據更新。
2. **空間複雜度：**
    
    - 使用固定長度的數組，空間複雜度為 O(n)O(n)O(n)，其中 nnn 是隊列的最大容量。

---

### **其他解法簡述**

1. **鏈表法：**
    
    - 使用雙向鏈表實現雙端隊列，每個節點存儲一個元素。
    - 插入與刪除操作均為 O(1)O(1)O(1)。
    - 空間複雜度為 O(n)O(n)O(n)。
2. **內建結構實現：**
    
    - 使用 Python 的 `collections.deque`，內建支持雙端操作，效率高且易用。
    - 插入與刪除操作為 O(1)O(1)O(1)，空間複雜度為 O(n)O(n)O(n)。

該解法基於循環數組，操作簡單高效，適合需要固定容量的場景。相比鏈表，內存分配更加緊湊，效率更高。