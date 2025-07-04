Lintcode 540
给你两个一维向量，实现一个迭代器，交替返回两个向量的元素

**样例1**
```python
输入: v1 = [1, 2] 和 v2 = [3, 4, 5, 6]
输出: [1, 3, 2, 4, 5, 6]
解释:
一开始轮换遍历两个数组，当v1数组遍历完后，就只遍历v2数组了，所以返回结果:[1, 3, 2, 4, 5, 6]
```
**样例2**
```python
输入: v1 = [1, 1, 1, 1] 和 v2 = [3, 4, 5, 6]
输出: [1, 3, 1, 4, 1, 5, 1, 6]
```



```python
class ZigzagIterator:

    def __init__(self, v1, v2):
        self.queue = [v for v in (v1, v2) if v]

    def next(self):
        v = self.queue.pop(0)
        value = v.pop(0)
        if v:
            self.queue.append(v)
        return value

    def hasNext(self):
        return len(self.queue) > 0

v1 = [1, 2]
v2 = [3, 4, 5, 6]
solution, result = ZigzagIterator(v1, v2), []
while solution.hasNext():
    result.append(solution.next())

print(result)
```
pass

v1 = [1, 2] 和 v2 = [3, 4, 5, 6]
解釋:
step1:
在constructer中queue = [ [1, 2], [3, 4, 5, 6] ]
step2:
solution.next() 
在next function裡面先用queue.pop()將v=[1,2]取出來, 
然後用value = v.pop(0) = 1 取出來, 所以solution.next() = 1
step3:
再把v=[2]存回queue = [ [3,4,5,6], [2] ]


如果是用Iterator
```python
"""
class ZigzagIterator:

    def __init__(self, v1, v2):
        self.queue = [v for v in (v1, v2) if v]

	def __iter__(self):
		return self

    def __next__(self):
        if not self.queue:
            raise StopIteration  # 當 queue 為空時，拋出 StopIteration 例外

        v = self.queue.pop(0)
        value = v.pop(0)
        if v:
            self.queue.append(v)
        return value


v1 = [1, 2]
v2 = [3, 4, 5, 6]

iterator = ZigzagIterator([v1, v2])

# Method1 - 不會調用__iter__ 只會直接使用 __next__
print(next(iterator))  # 1
print(next(iterator))  # 2

# Method2 - for num in iterator會調用__iter__ 之後使用 __next__
這意味著 `__iter__()` **在 `for` 迴圈開始時會自動被調用**，以確保 `iterator` 是可迭代的。
for num in iterator:
    print(num)

# Method3 - list會調用__iter__ 之後使用 __next__
這行代碼會調用 `__iter__()`，然後反覆調用 `__next__()`，直到 `StopIteration` 為止。

result = list(iterator)

```

`for` 迴圈的內部機制如下：
```python
iter_obj = iter(iterator)  # 這裡會調用 __iter__()
while True:
    try:
        num = next(iter_obj)  # 這裡會調用 __next__()
        print(num)
    except StopIteration:
        break
```


## **詳細解釋 `def __iter__(self): return self`**

### **1. `__iter__()` 是 Python 迭代器協議 (Iterator Protocol) 的一部分**

在 Python 中，**可迭代對象 (Iterable)** 是指 **可以被 `for` 迴圈 或 `iter()` 函數遍歷的對象**。根據 Python 迭代器協議，對象需要實作：

- `__iter__()`：返回一個 **迭代器對象 (Iterator)**。
- `__next__()`：返回序列中的下一個元素，並在沒有元素時拋出 `StopIteration`。

### **2. `__iter__(self): return self` 作用**

python

複製編輯

`def __iter__(self):     return self`

這段程式碼表示：

1. **`self` 既是可迭代對象 (Iterable)，也是一個迭代器 (Iterator)**。
2. 當 `iter(iterator)` 被調用時，它會返回 `self`，也就是說 `ZigzagIterator` 本身就是一個迭代器，不需要額外創建新的對象來管理狀態。

這種寫法的目的是讓 `ZigzagIterator` 既能：

- 被 `for` 迴圈直接遍歷 (`for num in iterator:`)
- 也能夠被 `iter()` 轉換為迭代器 (`iter(iterator)`)

---

## **`__iter__()` 何時會被調用？**

1. **當 `iter(iterator)` 被呼叫時**
    
    python
    
    複製編輯
    
    `iterator = ZigzagIterator(v1, v2) iter(iterator)  # 會調用 iterator.__iter__()`
    
    這會調用 `__iter__()`，並返回 `self`，因為 `ZigzagIterator` 本身就是一個迭代器。
    
2. **當 `for` 迴圈開始時**
    
    python
    
    複製編輯
    
    `for num in iterator:     print(num)`
    
    `for` 迴圈的內部機制如下：
    
    python
    
    複製編輯
    
    `iter_obj = iter(iterator)  # 這裡會調用 __iter__() while True:     try:         num = next(iter_obj)  # 這裡會調用 __next__()         print(num)     except StopIteration:         break`
    
    這意味著 `__iter__()` **在 `for` 迴圈開始時會自動被調用**，以確保 `iterator` 是可迭代的。
    
3. **當 `list(iterator)` 被呼叫時**
    
    python
    
    複製編輯
    
    `result = list(iterator)`
    
    這行代碼會調用 `__iter__()`，然後反覆調用 `__next__()`，直到 `StopIteration` 為止。
    

---

## **`result = list(iterator)` 是否合理？**

是 **合理的**，但需要注意一些細節。

### **1. `list(iterator)` 的運作方式**

`list(iterator)` 的作用：

1. **調用 `iter(iterator)`**：
    
    python
    
    複製編輯
    
    `iterator = ZigzagIterator(v1, v2) iter(iterator)  # 會調用 __iter__()`
    
    這裡 `__iter__()` 返回 `self`，所以 `list(iterator)` 會直接對 `iterator` 本身進行遍歷。
    
2. **反覆調用 `__next__()`**
    
    - `list(iterator)` 會**一直呼叫 `next(iterator)` 直到 `StopIteration` 為止**。
    - 這與 `for num in iterator:` 迴圈的運作方式相同。

### **2. `list(iterator)` 只能用一次**

因為 `ZigzagIterator` **本身就是一個迭代器**，它沒有內部機制來重置狀態，所以 `list(iterator)` 只能遍歷一次：

python

複製編輯

`iterator = ZigzagIterator([1, 3, 5], [2, 4, 6]) result1 = list(iterator)  # result1 = [1, 2, 3, 4, 5, 6]  result2 = list(iterator)  # result2 = []，因為 iterator 已經被消耗完`

當 `iterator` 被 `list()` 消耗完之後，它不會自動重置，因此第二次 `list(iterator)` 會返回空列表。

### **3. `list(iterator)` 的替代方案**

如果想讓 `ZigzagIterator` 支援多次迭代，可以修改 `__iter__()` 讓它返回一個新的 `ZigzagIterator` 實例：

python

複製編輯

`class ZigzagIterator:     def __init__(self, v1, v2):         self.v1 = list(v1)  # 保存原始數據         self.v2 = list(v2)         self.queue = [v for v in (self.v1, self.v2) if v]      def __iter__(self):         return ZigzagIterator(self.v1, self.v2)  # 返回一個新的實例      def __next__(self):         if not self.queue:             raise StopIteration         v = self.queue.pop(0)         value = v.pop(0)         if v:             self.queue.append(v)         return value`

這樣，每次 `iter(iterator)` 都會返回一個新的 `ZigzagIterator`，允許 `list(iterator)` 被多次使用：

python

複製編輯

`iterator = ZigzagIterator([1, 3, 5], [2, 4, 6]) result1 = list(iterator)  # [1, 2, 3, 4, 5, 6] result2 = list(iterator)  # [1, 2, 3, 4, 5, 6] (現在可以正確執行)`

---

## **總結**

1. **`__iter__()` 方法的作用**：
    
    - 讓 `ZigzagIterator` 變成可迭代對象。
    - `return self` 使 `ZigzagIterator` 既是 **可迭代對象 (Iterable)**，也是 **迭代器 (Iterator)**。
    - 當 `iter(iterator)`、`for` 迴圈 或 `list(iterator)` 被調用時，`__iter__()` 會被自動執行。
2. **`result = list(iterator)` 是否合理？**
    
    - **是合理的**，但 `ZigzagIterator` **只能被遍歷一次**，因為它是 **迭代器** 而不是可重用的可迭代對象。
    - 如果需要 **多次遍歷**，應該在 `__iter__()` 返回一個新的 `ZigzagIterator` 實例。

這樣，你就能清楚地理解 `__iter__()` 的作用、迭代器的工作方式，以及如何使用 `list(iterator)` 來收集所有元素！