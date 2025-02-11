Lintcode 494
通过两个队列实现一个栈。队列是先进先出(FIFO)。这意味着不能直接弹出队列中的最后一个元素。

### **LintCode 494 - 用双队列实现栈 解法詳細解釋**

---

### **題目分析**

使用兩個隊列來實現棧的功能，包括以下操作：

1. **`push(x)`**：將元素 `x` 推入棧頂。
2. **`pop()`**：移除並返回棧頂元素。
3. **`top()`**：返回棧頂元素，但不移除。
4. **`isEmpty()`**：判斷棧是否為空。

---

### **解法核心思路**

使用兩個隊列 `queue1` 和 `queue2`，其中：

1. **`queue1`**：作為主要存儲數據的隊列。
2. **`queue2`**：輔助隊列，用於將 `queue1` 中的元素按順序重新排列。

**關鍵點：**

- 當執行 `pop()` 或 `top()` 操作時，將 `queue1` 的所有元素（除最後一個）轉移到 `queue2`，最後一個元素即為棧頂元素。
- 操作完成後，交換兩個隊列的角色，使得 `queue1` 始終保持主存儲隊列。

```python
"""
输入：
push(1)
pop()
push(2)
isEmpty() // return false
top() // return 2
pop()
isEmpty() // return true
```
```python
输入：
isEmpty()
```

```python
from collections import deque
class Stack:
    
    def __init__(self):
        self.queue1 = deque()
        self.queue2 = deque()
        
    """
    @param: x: An integer
    @return: nothing
    """
    def push(self, x):
        self.queue1.append(x)

    """
    @return: nothing
    """
    def pop(self):
        for _ in range(len(self.queue1) - 1):
            val = self.queue1.popleft()
            self.queue2.append(val)
            
        val = self.queue1.popleft()
        self.queue1, self.queue2 = self.queue2, self.queue1
        return val

    """
    @return: An integer
    """
    def top(self):
        val = self.pop()
        self.push(val)
        return val

    """
    @return: True if the stack is empty
    """
    def isEmpty(self):
        return not self.queue1
```
pass

---

### **解法步驟**

#### **1. 初始化**

- 使用兩個雙端隊列 `deque`：
    - `queue1`：用於存儲棧內的元素。
    - `queue2`：作為輔助隊列。
- 初始時，兩個隊列均為空。

---

#### **2. 操作實現**

- **`push(x)`：**
    
    - 將元素 `x` 加入 `queue1` 的尾部。
    - 時間複雜度為 O(1)O(1)O(1)。
- **`pop()`：**
    
    1. 將 `queue1` 中的所有元素（除最後一個）逐一轉移到 `queue2`。
    2. 彈出並返回 `queue1` 中最後剩下的元素。
    3. 交換 `queue1` 和 `queue2`，使 `queue1` 成為主隊列。
    
    - 時間複雜度為 O(n)O(n)O(n)，其中 nnn 是棧的當前大小。
- **`top()`：**
    
    1. 調用 `pop()` 獲取棧頂元素。
    2. 再次將該元素壓回 `queue1`。
    
    - 時間複雜度為 O(n)O(n)O(n)。
- **`isEmpty()`：**
    
    - 檢查 `queue1` 是否為空。
    - 時間複雜度為 O(1)O(1)O(1)。

---

### **具體舉例**

#### 輸入：

python

複製程式碼

`push(1) pop() push(2) isEmpty() // return false top() // return 2 pop() isEmpty() // return true`

#### 步驟詳解：

1. **`push(1)`：**
    
    - 將 `1` 加入 `queue1`：
        - `queue1 = [1]`
        - `queue2 = []`
2. **`pop()`：**
    
    - 將 `queue1` 的所有元素（除最後一個）移動到 `queue2`：
        - `queue1 = []`
        - `queue2 = []`
    - 返回 `1`。
3. **`push(2)`：**
    
    - 將 `2` 加入 `queue1`：
        - `queue1 = [2]`
        - `queue2 = []`
4. **`isEmpty()`：**
    
    - 判斷 `queue1` 是否為空：
        - 返回 `False`。
5. **`top()`：**
    
    - 調用 `pop()` 獲取棧頂元素 `2`：
        - `queue1 = []`
        - `queue2 = []`
        - 返回 `2`。
    - 將 `2` 再次壓回 `queue1`：
        - `queue1 = [2]`
        - `queue2 = []`
6. **`pop()`：**
    
    - 將 `queue1` 的所有元素（除最後一個）移動到 `queue2`：
        - `queue1 = []`
        - `queue2 = []`
    - 返回 `2`。
7. **`isEmpty()`：**
    
    - 判斷 `queue1` 是否為空：
        - 返回 `True`。

---

### **時間與空間複雜度分析**

1. **時間複雜度：**
    
    - `push(x)`：O(1)O(1)O(1)。
    - `pop()`：O(n)O(n)O(n)，其中 nnn 是棧的當前大小。
    - `top()`：O(n)O(n)O(n)（需要調用 `pop()` 和 `push()`）。
    - `isEmpty()`：O(1)O(1)O(1)。
2. **空間複雜度：**
    
    - 需要兩個隊列，每個隊列的大小最多為 nnn，因此空間複雜度為 O(n)O(n)O(n)。

---

### **其他解法簡述**

1. **單隊列法：**
    
    - 只使用一個隊列模擬棧：
        - 在 `push(x)` 時，先將元素壓入隊列，然後將隊列中的其他元素依次移到隊尾，確保新元素始終在隊首。
    - **時間複雜度：**
        - `push(x)`：O(n)O(n)O(n)。
        - `pop()` 和 `top()`：O(1)O(1)O(1)。
2. **雙向鏈表法：**
    
    - 使用雙向鏈表直接模擬棧結構。
    - **時間複雜度：**
        - `push(x)`、`pop()` 和 `top()`：O(1)O(1)O(1)。

---

### **總結**

該解法通過兩個隊列實現棧操作，雖然 `push` 操作高效，但 `pop` 和 `top` 的時間複雜度為 O(n)O(n)O(n)。單隊列法在某些場景下效率更高，但實現稍微複雜。雙向鏈表法是另一個高效的選擇，操作簡單且效率穩定。