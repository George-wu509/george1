Lintcode 8
给定一个字符数组 `s` 和一个偏移量，根据偏移量**原地旋转**字符数组(从左向右旋转)。

**样例 1：**
输入：
```python
"""
s = "abcdefg"
offset = 3
```
输出：
```python
"""
"efgabcd"
```
解释
注意是**原地旋转**，即 s 旋转后为"efgabcd"

**样例 2：**
输入：
```python
"""
s = "abcdefg"
offset = 0
```
输出：
```python
"""
"abcdefg"
```
解释：
注意是**原地旋转**，即 s 旋转后为"abcdefg"

**样例 3：**
输入：
```python
"""
s = "abcdefg"
offset = 1
```
输出：
```python
"gabcdef"
```
解释：
注意是**原地旋转**，即 s 旋转后为"gabcdef"

**样例 4：**
输入：
```python
s = "abcdefg"
offset = 2
```
输出：
```python
"fgabcde"
```
解释：
注意是**原地旋转**，即 s 旋转后为"fgabcde"

**样例 5：**
输入：
```python
s = "abcdefg"
offset = 10
```
输出：
```python
"efgabcd"
```
解释：
注意是**原地旋转**，即 s 旋转后为"efgabcd"

最佳解法: 旋轉法
```python
class Solution:
    def rotateString(self, s, offset):
        if len(s) == 0:
            return

        offset = offset % len(s)  # 处理越界

        # 反转整个数组
        s.reverse()

        # 反转前 offset 个字符
        s[:offset] = reversed(s[:offset])

        # 反转剩余部分
        s[offset:] = reversed(s[offset:])
```


```python
class Solution:
    # @param s: a list of char
    # @param offset: an integer 
    # @return: nothing
    def rotateString(self, s, offset):
        if len(s) > 0:
            offset = offset % len(s)
            
        temp = (s + s)[len(s) - offset : 2 * len(s) - offset]

        for i in range(len(temp)):
            s[i] = temp[i]
```
pass

# **LintCode 8: Rotate String（旋转字符串）**

---

## **问题描述**

给定一个字符数组 `s` 和一个整数 `offset`，将 `s` **向右旋转 `offset` 位**（即后 `offset` 个字符移动到前面），并 **原地修改 `s`**。

---

## **示例**

`输入: s = ['a', 'b', 'c', 'd', 'e'] offset = 2  输出: s = ['d', 'e', 'a', 'b', 'c']`

**解释**

`初始: ['a', 'b', 'c', 'd', 'e'] 旋转 2 次: ['d', 'e', 'a', 'b', 'c']`

---

## **解法：拼接字符串**

### **核心思路**

1. **处理 `offset` 越界**
    
    - `offset` 可能大于 `s` 长度，只需要旋转 `offset % len(s)`： offset=offsetmod  len(s)\text{offset} = \text{offset} \mod \text{len(s)}offset=offsetmodlen(s)
    - 例如：

        `s = ['a', 'b', 'c', 'd', 'e'] offset = 7 offset = 7 % 5 = 2  # 只需旋转 2 位`
        
2. **利用 `s + s` 拼接**
    
    - **复制 `s` 两次**，形成 `s + s`，例如：

        `s = ['a', 'b', 'c', 'd', 'e'] s + s = ['a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e']`
        
    - **取出 `offset` 部分**

        `temp = (s + s)[len(s) - offset : 2 * len(s) - offset]`

        `temp = ['d', 'e', 'a', 'b', 'c']`
        
3. **赋值回 `s`**
    
    - 遍历 `temp`，逐个赋值回 `s[i]`，实现**原地修改**。

---

## **代码解析**
```python
class Solution:
    # @param s: a list of char
    # @param offset: an integer 
    # @return: nothing
    def rotateString(self, s, offset):
        if len(s) > 0:
            offset = offset % len(s)  # 处理 offset 过大情况
            
        temp = (s + s)[len(s) - offset : 2 * len(s) - offset]  # 获取旋转后的部分

        for i in range(len(temp)):  # 赋值回 s
            s[i] = temp[i]

```

---

## **执行过程**

`s = ['a', 'b', 'c', 'd', 'e'] offset = 2`

---

### **Step 1: 计算 `offset % len(s)`**

`offset = 2 % 5 = 2`

`offset` **小于 `s` 长度，无需变化**。

---

### **Step 2: 复制 `s` 并截取旋转部分**

`s + s = ['a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e'] temp = (s + s)[5 - 2 : 10 - 2] = (s + s)[3:8]`

得到：

`temp = ['d', 'e', 'a', 'b', 'c']`

---

### **Step 3: 赋值 `temp` 到 `s`**

`s[0] = 'd' s[1] = 'e' s[2] = 'a' s[3] = 'b' s[4] = 'c'`

最终 `s` 变为：

`s = ['d', 'e', 'a', 'b', 'c']`

---

## **时间与空间复杂度分析**

|操作|时间复杂度|空间复杂度|
|---|---|---|
|计算 `offset % len(s)`|`O(1)`|`O(1)`|
|复制 `s + s`|`O(n)`|`O(n)`|
|取 `temp = (s + s)[...]`|`O(n)`|`O(n)`|
|赋值 `s[i] = temp[i]`|`O(n)`|`O(1)`|
|**总复杂度**|`O(n)`|`O(n)`|

### **优化空间**

由于 `s + s` **额外使用 `O(n)` 空间**，可以用 **原地旋转** 方法优化 **空间到 `O(1)`**。

---

## **其他解法**

### **1. 反转法（最佳解法，`O(n)` 时间，`O(1)` 空间）**

- **思路**
    
    1. **反转整个数组**
    2. **反转前 `offset` 个字符**
    3. **反转剩余部分**
```python
s = ['a', 'b', 'c', 'd', 'e']
offset = 2
1. 反转整个数组: ['e', 'd', 'c', 'b', 'a']
2. 反转前 offset 个字符: ['d', 'e', 'c', 'b', 'a']
3. 反转剩余部分: ['d', 'e', 'a', 'b', 'c']

```
        
```python
class Solution:
    def rotateString(self, s, offset):
        if len(s) == 0:
            return

        offset = offset % len(s)  # 处理越界

        # 反转整个数组
        s.reverse()

        # 反转前 offset 个字符
        s[:offset] = reversed(s[:offset])

        # 反转剩余部分
        s[offset:] = reversed(s[offset:])

```

不用數組的reverse function:
```python
def reverse(s, start, end):
  """
  原地反轉列表 s 從索引 start 到 end (包含 end) 的部分
  """
  while start < end:
    s[start], s[end] = s[end], s[start]
    start += 1
    end -= 1

def rotate_string(s, offset):
  """
  原地向右旋轉字元陣列 s offset 位
  :param s: 字元列表
  :param offset: 偏移量
  """
  if not s or len(s) == 0 or offset is None:
    return # 處理空列表或無效輸入

  n = len(s)
  k = offset % n # 計算有效偏移量

  if k == 0:
    return # 如果偏移量是長度的倍數，則無需旋轉

  # 步驟 1: 反轉整個陣列
  reverse(s, 0, n - 1)
  # print(f"After step 1 (reverse all): {s}") # 用於除錯

  # 步驟 2: 反轉前 k 個元素
  reverse(s, 0, k - 1)
  # print(f"After step 2 (reverse first k): {s}") # 用於除錯

  # 步驟 3: 反轉剩下的 n-k 個元素
  reverse(s, k, n - 1)
  # print(f"After step 3 (reverse last n-k): {s}") # 用於除錯
```

---

### **2. 双指针滑动窗口**

- **思路**
    - 用 **双指针** 交换元素，从后往前逐个移动 `offset` 位。
- **复杂度**
    - 时间复杂度 `O(n)`
    - 空间复杂度 `O(1)`
```python
    class Solution:
    def rotateString(self, s, offset):
        if len(s) == 0:
            return

        offset = offset % len(s)
        for _ in range(offset):
            last = s[-1]
            for i in range(len(s) - 1, 0, -1):
                s[i] = s[i - 1]
            s[0] = last

```

---

## **方法比较**

|方法|思路|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|---|
|**拼接字符串（当前解法）**|`s + s` 取 `offset` 部分|`O(n)`|`O(n)`|**适用于不可修改 `s` 的情况**|
|**反转法（最佳解法）**|**三次反转** 实现旋转|`O(n)`|`O(1)`|**适用于大数据量 `s`**|
|**双指针滑动窗口**|**逐个后移 `offset`**|`O(n * offset)`|`O(1)`|**小规模 `offset` 适用**|

🚀 **反转法 (`O(n)` 时间, `O(1)` 空间) 是最优解！**





```python
    def rotateString1(self, s, offset):
        if len(s) == 0:
            return
        offset = offset % len(s)
        s.reverse()
        s[:offset] = reversed(s[:offset])
        s[offset:] = reversed(s[offset:])

    def rotateString2(self, s, offset):
        if len(s) == 0:
            return
	s = [s[-1-offset:-1], s[0:-2-offset]]
```

只有 **`rotateString1`** 符合原地旋转字符数组的要求。`rotateString2` 的实现**不是原地旋转**，而是创建了新的列表切片，并没有直接修改原始的字符数组 `s`。

**中文解释：**

**`rotateString1` 的解释：**

这个方法通过一系列巧妙的列表反转操作来实现原地旋转。让我们一步步分解：

1. **`if len(s) == 0: return`**: 如果字符数组为空，则无需旋转，直接返回。
    
2. **`offset = offset % len(s)`**: 这一步非常重要。它通过取偏移量 `offset` 对数组长度的模（余数），确保偏移量始终在 `[0, len(s) - 1]` 的范围内。例如，如果数组长度是 7，偏移量是 10，那么实际只需要旋转 10 % 7 = 3 位。这避免了不必要的多次完整旋转。
    
3. **`s.reverse()`**: 这一步将整个字符数组 `s` **原地反转**。例如，如果 `s` 是 `['a', 'b', 'c', 'd', 'e']`，反转后变成 `['e', 'd', 'c', 'b', 'a']`。
    
4. **`s[:offset] = reversed(s[:offset])`**: 这一步将数组的前 `offset` 个字符（需要移动到尾部的部分）进行**原地反转**。在上面的例子中，如果 `offset` 是 2，那么 `s[:2]` 是 `['e', 'd']`，反转后变成 `['d', 'e']`。此时 `s` 变成了 `['d', 'e', 'c', 'b', 'a']`。
    
5. **`s[offset:] = reversed(s[offset:])`**: 这一步将数组从索引 `offset` 开始到末尾的字符（原本在数组前部的部分，现在在中间）进行**原地反转**。在上面的例子中，`s[2:]` 是 `['c', 'b', 'a']`，反转后变成 `['a', 'b', 'c']`。最终，`s` 变成了 `['d', 'e', 'a', 'b', 'c']`，这正是将原始数组 `['a', 'b', 'c', 'd', 'e']` 向右旋转 2 位的结果。
    

**关键在于，`s.reverse()` 和对切片使用 `reversed()` 并赋值回原切片 (`s[:offset] = ...`, `s[offset:] = ...`) 都是在原始的 `s` 列表上进行修改，实现了原地操作。**

**`rotateString2` 的解释：**

1. **`if len(s) == 0: return`**: 如果字符数组为空，则无需旋转，直接返回。
    
2. **`s = [s[-1-offset:-1], s[0:-2-offset]]`**: 这一行代码的问题在于它**创建了包含两个新列表的列表，并重新赋值给了 `s`**。
    
    - `s[-1-offset:-1]`: 这部分尝试从 `s` 的尾部截取一部分元素。
    - `s[0:-2-offset]`: 这部分尝试从 `s` 的头部截取一部分元素。
    - **最关键的是，`s = [...]` 这一步创建了一个全新的列表，而不是在原来的 `s` 列表上进行修改。** 因此，这不是原地旋转。

**为什么 `rotateString2` 不符合原地旋转的要求：**

原地旋转意味着在不创建额外数据结构（在空间复杂度上通常要求 O(1) 的额外空间）的情况下，直接修改输入的数组。`rotateString2` 通过列表切片创建了新的列表，并用这个新列表替换了原来的 `s`。这改变了 `s` 的引用，但并没有在原始的内存空间中进行元素的移动和修改。

**总结：**

`rotateString1` 通过一系列的反转操作，巧妙地在原始的字符数组 `s` 上进行了修改，实现了原地旋转。`rotateString2` 则创建了新的列表，并没有满足原地旋转的要求。因此，**`rotateString1` 是符合题目要求的解法。**