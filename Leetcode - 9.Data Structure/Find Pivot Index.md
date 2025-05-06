Lintcode 1068
给定一个整数数组`nums`，编写一个返回此数组的“中心索引”的方法。
我们将中心索引定义为：中心索引左边的数字之和等于中心索引右边的数字之和。
如果不存在这样的中心索引，我们应该返回-1。 如果有多个中心索引，则应返回最左侧的那个。

**样例1:**
```python
"""
输入: 
nums = [1, 7, 3, 6, 5, 6]
输出: 3
解释: 
索引3 (nums[3] = 6)左侧所有数之和等于右侧之和。
并且3是满足条件的第一个索引。
```
**样例2:**
```python
"""
输入: 
nums = [1, 2, 3]
输出: -1
解释: 
并没有满足条件的中心索引。
```


```python
def pivot_index(self, nums: List[int]) -> int:
	total = sum(nums)
	_sum = 0
	for i, n in enumerate(nums):
		if _sum * 2 + n == total: 
		    return i
		_sum += n
	return -1
```
pass
解釋:
step1: 用sum(nums)求數組總合
step2: for loop累加n 並存入_sum. 
step3: 直到遇到這個n+ __ sum  等於一半的數組總和, 停下來並return i

## **LintCode 1068: 寻找数组的中心索引（Pivot Index）**

---

### **题目描述**

给定一个整数数组 `nums`，找出 **中心索引 `i`**，满足：

左侧元素和=右侧元素和\text{左侧元素和} = \text{右侧元素和}左侧元素和=右侧元素和

即：

∑0→i−1nums[j]=∑i+1→n−1nums[j]\sum_{0 \to i-1} nums[j] = \sum_{i+1 \to n-1} nums[j]0→i−1∑​nums[j]=i+1→n−1∑​nums[j]

如果 **不存在中心索引**，返回 `-1`。

#### **示例**

`输入： nums = [1, 7, 3, 6, 5, 6] 输出： 3`

**解释**

``索引 3 是中心索引，因为左侧元素和 `1 + 7 + 3 = 11`， 右侧元素和 `5 + 6 = 11`，两者相等。``

---

## **解法：前缀和 + 双指针**

### **核心思路**

1. **利用总和 `total`**
    
    - 设 `total = sum(nums)` 是整个数组的和。
    - **如果索引 `i` 是中心索引**，那么： 2×左侧和+nums[i]=total2 \times \text{左侧和} + \text{nums}[i] = \text{total}2×左侧和+nums[i]=total
    - **推导**： 左侧和=total−nums[i]2\text{左侧和} = \frac{\text{total} - \text{nums}[i]}{2}左侧和=2total−nums[i]​
2. **维护 `左侧前缀和 _sum`**
    
    - 遍历数组，如果 `2 * _sum + nums[i] == total`，说明 `i` 是中心索引。

---

### **代码解析**
```python
def pivot_index(self, nums: List[int]) -> int:
    total = sum(nums)  # 计算总和
    _sum = 0  # 维护左侧前缀和

    for i, n in enumerate(nums):
        if _sum * 2 + n == total:  # 检查中心索引条件
            return i
        _sum += n  # 更新左侧前缀和

    return -1  # 没有找到中心索引

```

---

## **逐步执行分析**

**输入**

python

複製編輯

`nums = [1, 7, 3, 6, 5, 6]`

### **初始化**

|变量|初始值|
|---|---|
|`total`|`1 + 7 + 3 + 6 + 5 + 6 = 28`|
|`_sum`|`0`|

---

### **遍历 `nums`**

|`i`|`n = nums[i]`|`2 * _sum + n`|`total`|判断|`_sum` 变化|
|---|---|---|---|---|---|
|0|1|`2 * 0 + 1 = 1`|`28`|❌|`_sum = 1`|
|1|7|`2 * 1 + 7 = 9`|`28`|❌|`_sum = 8`|
|2|3|`2 * 8 + 3 = 19`|`28`|❌|`_sum = 11`|
|3|6|`2 * 11 + 6 = 28`|`28`|✅ **返回 `3`**||

最终返回：

python

複製編輯

`3`

---

## **时间与空间复杂度分析**

1. **时间复杂度**
    
    - **`O(n)`**：遍历 `nums` 一次，每次 `O(1)` 计算 `_sum`。
2. **空间复杂度**
    
    - **`O(1)`**：只使用了常数个变量。

---

## **其他解法**

### **1. 前缀和 `O(n)`**

- 计算 `left_sum` 和 `right_sum` 两个数组。
- **时间 `O(n)`，空间 `O(n)`**。

```python
def pivot_index(nums):
    n = len(nums)
    if n == 0: return -1

    left_sum = [0] * n
    right_sum = [0] * n

    # 计算左侧和
    for i in range(1, n):
        left_sum[i] = left_sum[i - 1] + nums[i - 1]

    # 计算右侧和
    for i in range(n - 2, -1, -1):
        right_sum[i] = right_sum[i + 1] + nums[i + 1]

    # 找到满足条件的索引
    for i in range(n):
        if left_sum[i] == right_sum[i]:
            return i

    return -1

```

---

## **方法比较**

|方法|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|
|**前缀和 + 双指针（最佳解法）**|`O(n)`|`O(1)`|**最优**|
|前缀和数组|`O(n)`|`O(n)`|**适用于查找多个索引**|