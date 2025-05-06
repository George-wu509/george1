Lintcode 607
设计并实现一个 TwoSum 类。他需要支持以下操作:`add` 和 `find`。  
`add` -把这个数添加到内部的数据结构。  
`find` -是否存在任意一对数字之和等于这个值

Example:
```python
add(1);add(3);add(5);
find(4)//返回true
find(7)//返回false
```


```python
class TwoSum(object):

    def __init__(self):
        # initialize your data structure here
        self.count = {}
        
    # Add the number to an internal data structure.
    # @param number {int}
    # @return nothing
    def add(self, number):
        if number in self.count:
            self.count[number] += 1
        else:
            self.count[number] = 1

    # Find if there exists any pair of numbers which sum is equal to the value.
    # @param value {int}
    # @return true if can be found or false
    def find(self, value):
        for num in self.count:
            if value - num in self.count and \
                (value - num != num or self.count[num] > 1):
                return True
        return False
```
pass
解釋: 
step1: create 一個dict() counter儲存add()加入的值, 譬如{1:2, 2:1, 4:1}
step2: 遍歷這個counter (1->2->4), 然後根據要查找的target找(查找key = target-目前key). 如果有則代表可以找到兩個數和為target


# **LintCode 607: Two Sum III（数据结构设计）**

---

## **问题描述**

实现一个数据结构，支持以下操作：

1. **`add(number)`** - 向数据结构中添加一个数 `number`。
2. **`find(value)`** - 查找是否存在两个数的和等于 `value`。

---

## **解法：哈希表**

### **核心思路**

- **使用字典 `self.count` 存储每个数及其出现次数**，以 **O(1)** 的时间复杂度进行插入和查找。
- **对于 `find(value)`，遍历 `self.count`，检查 `value - num` 是否在 `self.count` 中**：
    - 如果 `value - num` 存在：
        - 如果 `value - num` **与 `num` 不同**，直接返回 `True`。
        - 如果 `value - num` **与 `num` 相同**，需要确保 `num` 出现 **至少两次** 才能满足条件。

---

## **执行过程**

### **变量表**

|变量|说明|
|---|---|
|`self.count`|存储数值及其出现次数的哈希表|

---

### **Step 1: `add()` 操作**

假设进行以下 `add` 操作：

`add(1) add(3) add(5) add(3)`

`self.count` 变化：

|操作|`self.count`|
|---|---|
|`add(1)`|`{1: 1}`|
|`add(3)`|`{1: 1, 3: 1}`|
|`add(5)`|`{1: 1, 3: 1, 5: 1}`|
|`add(3)`|`{1: 1, 3: 2, 5: 1}`|

---

### **Step 2: `find()` 操作**

假设我们执行 `find(4)`：

- 遍历 `self.count`：
    - `num = 1`，`4 - 1 = 3` 存在于 `self.count`，**返回 `True`**。

执行 `find(6)`：

- 遍历 `self.count`：
    - `num = 1`，`6 - 1 = 5` 存在于 `self.count`，**返回 `True`**。

执行 `find(7)`：

- 遍历 `self.count`：
    - `num = 1`，`7 - 1 = 6` **不存在**。
    - `num = 3`，`7 - 3 = 4` **不存在**。
    - `num = 5`，`7 - 5 = 2` **不存在**。
- **返回 `False`**。

---

## **时间与空间复杂度分析**

### **时间复杂度**

|操作|复杂度|说明|
|---|---|---|
|`add(number)`|`O(1)`|直接插入哈希表|
|`find(value)`|`O(n)`|遍历 `self.count`，检查 `value - num` 是否存在|

### **空间复杂度**

- 需要存储 **所有插入的数字**，**`O(n)`**。

---

## **其他解法**

### **1. 使用 `set` 记录所有可能的和**

- **思路**
    - `add(number)` 时，直接计算 **所有可能的 `a + b` 并存入 `set`**。
    - `find(value)` 时，只需 `O(1)` 查询 `set` 是否包含 `value`。
- **时间复杂度**
    - `add(number)`: `O(n)`
    - `find(value)`: `O(1)`

---

### **2. 使用排序 + 双指针**

- **思路**
    - `add(number)` 时，保持数组 **有序插入**。
    - `find(value)` 时，使用 **双指针 `O(n)`** 查找。
- **时间复杂度**
    - `add(number)`: `O(log n)`
    - `find(value)`: `O(n)`

---

## **方法比较**

|方法|`add()` 复杂度|`find()` 复杂度|适用情况|
|---|---|---|---|
|**哈希表（当前解法）**|`O(1)`|`O(n)`|**适用于大量 `add()` 操作**|
|**`set` 记录所有和**|`O(n)`|`O(1)`|**适用于大量 `find()` 操作**|
|**排序 + 双指针**|`O(log n)`|`O(n)`|**适用于 `find()` 操作远多于 `add()`**|

---

## **总结**

- **当前解法 `O(1)` 添加，`O(n)` 查找**，适合 **`add()` 操作多** 的情况。
- **如果 `find()` 频率远高于 `add()`，可使用 `set` 预计算和**，使 `find()` 降为 `O(1)`。
- **如果 `add()` 和 `find()` 频率接近，可以用 **排序 + 双指针 `O(n)`** 查找。

🚀 **当前解法最均衡，适用于通用场景！**







