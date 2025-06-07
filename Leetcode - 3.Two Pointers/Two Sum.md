Lintcode 56
给一个整数数组，找到两个数使得他们的和等于一个给定的数 `target`。

你需要实现的函数`twoSum`需要返回这两个数的下标, 并且第一个下标小于第二个下标。注意这里下标的范围是 `0` 到 `n-1`。

Example
```python
样例 1：
输入：
numbers = [2,7,11,15]
target = 9
输出：
[0,1]
解释：
numbers[0] + numbers[1] = 9

样例 2：
输入：
numbers = [15,2,7,11]
target = 9
输出：
[1,2]
解释：
numbers[1] + numbers[2] = 9
```


```python
def twoSum(self, numbers, target):
	if not numbers:
		return [-1, -1]
	
	# transform numbers to a sorted array with index
	nums = [
		(number, index)
		for index, number in enumerate(numbers)]
	nums = sorted(nums)
	
	left, right = 0, len(nums) - 1
	while left < right:
		if nums[left][0] + nums[right][0] > target:
			right -= 1
		elif nums[left][0] + nums[right][0] < target:
			left += 1
		else:
			return sorted([nums[left][1], nums[right][1]])
	
	return [-1, -1]
```
pass

排序的兩種方法
nums = **sorted**(nums)
nums.**sort**()
這裡不能用return [ nums[left][1], nums[right][1] ].sort() 只能用return sorted([nums[left][1], nums[right][1]]) 因為sort()只會更新nums, 但return None
在code裡面nums = sorted(nums) 可以改成nums.sort() 沒問題


## **问题描述**

给定一个整数数组 `numbers` 和一个整数 `target`，找出数组中**两个数的索引**，使得它们的和等于 `target`。  
返回这两个数的索引，**索引需按升序排序**。如果没有符合条件的数对，返回 `[-1, -1]`。

---

## **示例**

`输入: numbers = [2, 7, 11, 15] target = 9  输出: [0, 1]`

**解释**

`2 + 7 = 9，因此返回 [0,1]`

---

## **解法：双指针（排序 + 双指针扫描）**

### **核心思路**

1. **转换数组**：由于原数组索引不能丢失，我们需要 **同时存储元素值和索引**：

    `nums = [(number, index) for index, number in enumerate(numbers)]`
    
    **示例**

    `输入: numbers = [2, 7, 11, 15] 转换: [(2, 0), (7, 1), (11, 2), (15, 3)]`
    
2. **排序数组**：
    
    - 由于 `Two Sum` 需要找 `a + b = target`，我们可以 **先排序数组**，然后使用 **双指针** 查找答案：

    `nums = sorted(nums)`
    
    **示例**

    `排序后: [(2, 0), (7, 1), (11, 2), (15, 3)]`
    
3. **使用双指针 `left` 和 `right`**：
    
    - **`left` 指向数组头部（最小值）**
    - **`right` 指向数组尾部（最大值）**
    - **比较 `nums[left][0] + nums[right][0]` 与 `target`**：
        - **如果和大于 `target`**，右指针左移（`right -= 1`）
        - **如果和小于 `target`**，左指针右移（`left += 1`）
        - **如果和等于 `target`**，返回 `[nums[left][1], nums[right][1]]`。

---

## **代码解析**
```python
"""
def twoSum(self, numbers, target):
    if not numbers:
        return [-1, -1]

    # 将数组转换为 (数值, 索引) 对，并排序
    nums = [(number, index) for index, number in enumerate(numbers)]
    nums = sorted(nums)

    left, right = 0, len(nums) - 1

    while left < right:
        if nums[left][0] + nums[right][0] > target:
            right -= 1  # 总和过大，移动右指针
        elif nums[left][0] + nums[right][0] < target:
            left += 1  # 总和过小，移动左指针
        else:
            return sorted([nums[left][1], nums[right][1]])  # 返回索引并排序
    
    return [-1, -1]  # 未找到结果

```
---

## **执行过程**

**输入**

`numbers = [2, 7, 11, 15] target = 9`

---

### **Step 1: 转换数组**

`nums = [(2, 0), (7, 1), (11, 2), (15, 3)]`

**存储索引，避免排序后丢失索引信息**。

---

### **Step 2: 排序数组**

`sorted(nums) = [(2, 0), (7, 1), (11, 2), (15, 3)]`

排序后仍是相同顺序（原数组已排序）。

---

### **Step 3: 双指针查找**

#### **初始状态**

`left = 0, right = 3 nums[left] = (2, 0), nums[right] = (15, 3)`

#### **Step 3.1: 第一次循环**

`nums[left][0] + nums[right][0] = 2 + 15 = 17 大于 target = 9，移动右指针`

`right = 2`

#### **Step 3.2: 第二次循环**

`nums[left][0] + nums[right][0] = 2 + 11 = 13 仍然大于 target = 9，移动右指针`

`right = 1`

#### **Step 3.3: 第三次循环**

`nums[left][0] + nums[right][0] = 2 + 7 = 9 找到目标值，返回 [0, 1]`

**最终返回**

`[0, 1]`

---

## **时间与空间复杂度分析**

|操作|时间复杂度|空间复杂度|
|---|---|---|
|**构建 (数值, 索引) 对**|`O(n)`|`O(n)`|
|**排序数组**|`O(n log n)`|`O(n)`|
|**双指针扫描**|`O(n)`|`O(1)`|
|**总复杂度**|`O(n log n)`|`O(n)`|

### **优化空间**

如果 **不考虑返回索引顺序**，可以使用 **哈希表解法**，将 **空间优化到 `O(n)`**。

---

## **其他解法**

### **1. 哈希表解法（`O(n)` 时间，`O(n)` 空间）**

- **思路**
    - 用 **哈希表（字典）存储 `number -> index`**。
    - 遍历 `numbers`，检查 `target - number` 是否在哈希表中。

---

### **2. 直接暴力法（`O(n^2)`）**

- **思路**
    - **双重循环** 遍历所有可能的 `(i, j)` 组合。
    - **若 `numbers[i] + numbers[j] == target`，返回 `[i, j]`**。
    - **时间复杂度 `O(n^2)`**，不适用于大规模数据。

---

### **3. 二分查找（`O(n log n)`）**

- **思路**
    - **先排序 `numbers`**
    - **遍历 `numbers[i]`，用二分查找 `target - numbers[i]`**
    - **时间复杂度 `O(n log n)`**。

---

## **方法比较**

|方法|思路|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|---|
|**双指针（当前解法）**|**排序 + 双指针扫描**|`O(n log n)`|`O(n)`|**适用于可变数组**|
|**哈希表**|**存储已遍历值，加速查找**|`O(n)`|`O(n)`|**最优解，适用于大数据**|
|**暴力法**|**双层循环遍历所有组合**|`O(n^2)`|`O(1)`|**数据较小时可用**|
|**二分查找**|**排序 + 二分查找**|`O(n log n)`|`O(n)`|**适用于已排序数组**|

---

## **总结**

- **最佳解法** ✅ **哈希表 `O(n)` 时间 `O(n)` 空间**
- **双指针 `O(n log n)` 适用于不想额外用 `O(n)` 空间**
- **暴力法 `O(n^2)` 仅适用于小规模数据**
- **二分查找 `O(n log n)` 适用于已排序数组**

🚀 **哈希表是最优解，但双指针适用于不想额外用 `O(n)` 空间的情况！**
