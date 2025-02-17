intcode 12
实现一个栈, 支持以下操作:

- `push(val)` 将 val 压入栈
- `pop()` 将栈顶元素弹出, 并返回这个弹出的元素
- `min()` 返回栈中元素的最小值

要求 O(1) 开销

Example:

输入：
push(1)
min()
push(2)
min()
push(3)
min()
输出：
1
1
1

```python
class MinStack:
 
    def __init__(self):
        self.stack = []
        self.min_stack = []
 
    """
    @param: number: An integer
    @return: nothing
    """
    def push(self, number):
        self.stack.append(number)
        if len(self.min_stack) == 0 or number < self.min_stack[len(self.min_stack) - 1]:
            self.min_stack.append(number)
        else:
            self.min_stack.append(self.min_stack[len(self.min_stack) - 1])
 
    """
    @return: An integer
    """
    def pop(self):
        self.min_stack.pop()
        return self.stack.pop()
 
    """
    @return: An integer
    """
    def min(self):
        return self.min_stack[len(self.min_stack) - 1]
```
pass


### **解題思路：Min Stack (最小棧)**

Min Stack 是一個支援 `push(x)`, `pop()`, `top()`, `getMin()` 四種操作的資料結構，其中 `getMin()` 需在 **常數時間 O(1)** 內完成。  
一般來說，普通的棧（Stack）只能支援 `push`、`pop`、`top`，但無法高效實現 `getMin()`，因為 `getMin()` 需要額外搜尋整個棧來找出最小值，這樣會變成 **O(n)**。

我們可以用 **輔助棧 (minStack)** 來記錄當前棧的最小值，使 `getMin()` 操作變成 **O(1)**。

---

### **解法: 使用輔助棧**

我們維護兩個棧：

1. **主棧 (stack)**：用於正常存儲元素
2. **最小值棧 (minStack)**：用於存儲每一步的最小值

#### **主要操作邏輯**

|操作|說明|
|---|---|
|`push(x)`|把 x 放入主棧，若 x 小於等於 minStack 的頂部，則同步壓入 minStack|
|`pop()`|從主棧彈出元素，若該元素等於 minStack 的頂部，則同步彈出 minStack|
|`top()`|返回主棧的頂部元素|
|`getMin()`|返回 minStack 的頂部，即為當前棧中的最小值|

#### **範例**

假設我們依次執行以下操作：

python

複製編輯

`push(3) push(5) push(2) push(1) pop() getMin()`

|操作|主棧 (stack)|最小值棧 (minStack)|`getMin()`|
|---|---|---|---|
|push(3)|[3]|[3]|3|
|push(5)|[3, 5]|[3]|3|
|push(2)|[3, 5, 2]|[3, 2]|2|
|push(1)|[3, 5, 2, 1]|[3, 2, 1]|1|
|pop()|[3, 5, 2]|[3, 2]|2|
|getMin()|[3, 5, 2]|[3, 2]|**2**|

> `pop()` 操作時，如果彈出的元素恰好是最小值棧的頂部，那麼 minStack 也要同步彈出，確保最小值正確更新。

---

### **時間與空間複雜度分析**

|操作|時間複雜度|空間複雜度|
|---|---|---|
|`push(x)`|O(1)|O(1)|
|`pop()`|O(1)|O(1)|
|`top()`|O(1)|O(1)|
|`getMin()`|O(1)|O(1)|

- **時間複雜度**：
    - 每個操作都只涉及 **棧的頂部元素**，因此所有操作都是 **O(1)**。
- **空間複雜度**：
    - `minStack` 只存儲 **不遞增** 的元素，因此在最壞情況下，minStack 會和主棧一樣大，額外空間為 **O(n)**。

---

### **其他解法思路**

1. **使用單棧與 Pair**
    
    - 棧中的每個元素存 `(x, minValue)`，即當前值與當前最小值。
    - `getMin()` 直接讀取頂部的 `minValue`，但空間複雜度仍是 **O(n)**。
2. **單棧壓縮版**
    
    - 只在 `x <= minStack.top()` 時存入 `minStack`，減少額外空間使用。
    - `pop()` 時，若彈出元素與 `minStack.top()` 相同，則 `minStack` 也要彈出。
3. **數組模擬棧**
    
    - 直接用陣列模擬棧，每次 `getMin()` 時遍歷陣列 **O(n)**，但不需要額外空間。