Lintcode 976
给出 A, B, C, D 四个整数列表，计算有多少的tuple `(i, j, k, l)`满足`A[i] + B[j] + C[k] + D[l]`为 0。

为了简化问题，A, B, C, D 具有相同的长度，且长度N满足 0 ≤ N ≤ 500。所有的整数都在范围(-2^28, 2^28 - 1)内以及保证结果最多为2^31 - 1。

Example:
**样例 1：**
输入：
```python
"""
numbers = [2,7,11,15]
target = 3
```
输出：
```python
"""
[]
```
解释：
2 + 7 + 11 + 15 ！= 3，不存在满足条件的四元组。

**样例 2：**
输入：
```python
"""
numbers = [1,0,-1,0,-2,2]
target = 0
```
输出：
```python
"""
[[-1, 0, 0, 1],[-2, -1, 1, 2],[-2, 0, 0, 2]]
```
解释：
有3个不同的四元组满足四个数之和为0。

```python
def four_sum(self, nums, target):
	nums.sort()
	res = []
	length = len(nums)
	for i in range(0, length - 3):
		if i and nums[i] == nums[i - 1]:
			continue
		for j in range(i + 1, length - 2):
			if j != i + 1 and nums[j] == nums[j - 1]:
				continue
			sum = target - nums[i] - nums[j]
			left, right = j + 1, length - 1
			while left < right:
				if nums[left] + nums[right] == sum:
					res.append([nums[i], nums[j], nums[left], nums[right]])
					right -= 1
					left += 1
					while left < right and nums[left] == nums[left - 1]:
						left += 1
					while left < right and nums[right] == nums[right + 1]:
						right -= 1
				elif nums[left] + nums[right] > sum:
					right -= 1
				else:
					left += 1
	return res
```
pass


# **LintCode 976: 4Sum II（四数之和 II）**

---

## **问题描述**

给定四个整数数组 `A, B, C, D`，计算 **有多少个四元组 `(a, b, c, d)`** 满足：

A[i]+B[j]+C[k]+D[l]=0A[i] + B[j] + C[k] + D[l] = 0A[i]+B[j]+C[k]+D[l]=0

即 **四个数组中选一个数，使得它们的和等于 0**。

---

## **解法：哈希表（两数和 + 查找）**

### **核心思路**

1. **使用哈希表存储 `A[i] + B[j]` 的所有可能值**
    
    - 遍历 `A, B`，计算 `A[i] + B[j]`，存入 `hash_map`，并记录**出现次数**。
2. **遍历 `C[k] + D[l]`，查找 `-(C[k] + D[l])` 是否在 `hash_map`**
    
    - 如果 `hash_map` 里有 `-(C[k] + D[l])`，说明找到了一个符合条件的四元组。

---

## **执行过程**

### **变量表**

|变量|说明|
|---|---|
|`hash_map`|存储 `A[i] + B[j]` 的出现次数|
|`count`|记录满足 `A[i] + B[j] + C[k] + D[l] = 0` 的四元组个数|

---

### **Step 1: 构建哈希表**

假设：

python

複製編輯

`A = [1, 2] B = [-2, -1] C = [-1, 2] D = [0, 2]`

计算 `A[i] + B[j]`，存入 `hash_map`：

複製編輯

`(1) 1 + (-2) = -1   → hash_map[-1] = 1 (2) 1 + (-1) =  0   → hash_map[0]  = 1 (3) 2 + (-2) =  0   → hash_map[0]  = 2 (4) 2 + (-1) =  1   → hash_map[1]  = 1`

最终 `hash_map`：

yaml

複製編輯

`{-1: 1, 0: 2, 1: 1}`

---

### **Step 2: 查找 `-(C[k] + D[l])`**

计算 `C[k] + D[l]`，查找 `-(C[k] + D[l])`：

go

複製編輯

``(1) -(-1 + 0) =  1   → 在 hash_map 中找到 `1`（1 次） (2) -(-1 + 2) = -1   → 在 hash_map 中找到 `-1`（1 次） (3) -( 2 + 0) = -2   → 不在 hash_map (4) -( 2 + 2) = -4   → 不在 hash_map``

满足条件的四元组数 `count = 2`。

---

### **最终结果**

python

複製編輯

`count = 2`

返回：

python

複製編輯

`2`

---

## **时间与空间复杂度分析**

### **时间复杂度**

|操作|复杂度|说明|
|---|---|---|
|**计算 `A[i] + B[j]` 存入 `hash_map`**|`O(n²)`|两层循环遍历 `A, B`|
|**遍历 `C[k] + D[l]` 查找 `hash_map`**|`O(n²)`|两层循环遍历 `C, D`|
|**总复杂度**|`O(n²)`|由于两层 `O(n²)` 迭代，最终 `O(n²)`|

### **空间复杂度**

- 需要 `O(n²)` 额外空间存 `hash_map`。

---

## **其他解法**

### **1. 四重循环（暴力解法 O(n⁴)）**

- **思路**
    - 直接遍历 `A, B, C, D`，暴力计算所有 `A[i] + B[j] + C[k] + D[l]`。
- **时间复杂度**
    - **`O(n⁴)`**，适用于 `n` 较小时。

---

### **2. 排序 + 双指针（O(n³)）**

- **思路**
    - 对 `C + D` 排序后，用双指针查找 `-(A[i] + B[j])`。
- **时间复杂度**
    - **`O(n³)`**，适用于 `n` 适中。

---

## **方法比较**

|方法|思路|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|---|
|**哈希表（当前解法）**|**存 `A+B`，查找 `-(C+D)`**|`O(n²)`|`O(n²)`|**最优解，适用于大数据**|
|**暴力解法**|**四重循环遍历所有组合**|`O(n⁴)`|`O(1)`|**适用于小数据**|
|**排序 + 双指针**|**排序 `C+D`，查找 `-(A+B)`**|`O(n³)`|`O(1)`|**适用于 `n` 适中**|

---

## **总结**

- **最优解** ✅ **哈希表 `O(n²)`**
- **如果 `n` 较小，可用 `O(n⁴)` 暴力解法**
- **如果 `n` 适中，可用 `O(n³)` 排序 + 双指针**