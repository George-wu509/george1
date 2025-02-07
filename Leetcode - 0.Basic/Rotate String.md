Lintcode 8
给定一个字符数组 `s` 和一个偏移量，根据偏移量**原地旋转**字符数组(从左向右旋转)。

**样例 1：**
输入：
```python
s = "abcdefg"
offset = 3
```
输出：
```python
"efgabcd"
```
解释
注意是**原地旋转**，即 s 旋转后为"efgabcd"

**样例 2：**
输入：
```python
s = "abcdefg"
offset = 0
```
输出：
```python
"abcdefg"
```
解释：
注意是**原地旋转**，即 s 旋转后为"abcdefg"

**样例 3：**
输入：
```python
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