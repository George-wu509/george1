Lintcode 1844
给定一个整数数组和一个整数k，你需要找到和为k的最短**非空**子数组，并返回它的长度。
如果没有这样的子数组，返回-1.
**样例1：**
```python
输入: 
nums = [1,1,1,2] and k = 3
输出: 
2
```
**样例2：**
```python
输入: 
nums = [2,1,-1,4,2,-3] and k = 3
输出: 
2
```


```python
def subarray_sum_equals_k_i_i(self, nums: List[int], k: int) -> int:
	# 前缀和列表。
	sums = [0]
	# 保存前缀和及数组索引的 dict。
	hash_map = {0: 0}
	res = -1
	# 计算前缀和的同时计算满足题意的非空子列表。
	for i in range(len(nums)):
		n = nums[i]
		# 前缀和。
		prefix_sum = sums[len(sums) - 1] + n
		# 前缀和放入 sums 列表中。
		sums.append(prefix_sum)
		# 查找满足当前前缀和 sum - k 的距离当前位置最近的前缀和。
		idx = hash_map.get(prefix_sum - k)
		# 如果找到，则更新当前的结果。
		if idx is not None:
			length = i - idx + 1
			if res < 0 or length < res:
				res = length
		# 将当前前缀和与数组索引放到哈希表中。
		hash_map[prefix_sum] = i + 1
	return res
```
pass

解釋:
step1

# **LintCode 1844: Subarray Sum Equals to K II（最短子数组和等于 K）**

---

## **题目描述**

给定一个整数数组 `nums` 和一个目标值 `k`，找到 **和等于 `k` 的最短非空子数组**，返回其**长度**。

如果 **不存在这样的子数组**，返回 `-1`。

#### **示例**

`输入： nums = [1, 1, 1, 2, 3, 4] k = 5  输出： 2`

**解释**

`满足条件的子数组有： [2,3]（长度=2） [1,4]（长度=2）`

最短的长度是 `2`，所以返回 `2`。

---

## **解法：前缀和 + 哈希表（双指针）**

### **核心思路**

1. **利用前缀和**
    
    - 计算 `prefix_sum[i]`，表示 `nums[0]` 到 `nums[i]` 的累积和： 
    - 若 **前缀和 `prefix_sum[i] - k` 之前出现过**，则说明**存在 `sum(nums[l+1:i]) = k`**。
2. **哈希表存储前缀和**
    
    - 维护 `hash_map`（字典）：记录**某个前缀和首次出现的索引**。
    - **如果 `prefix_sum - k` 在 `hash_map` 中，计算当前子数组长度**，更新 `res`。

---

## **代码解析**
```python
def subarray_sum_equals_k_i_i(self, nums: List[int], k: int) -> int:
    # 前缀和列表
    sums = [0]
    # 记录前缀和首次出现的索引
    hash_map = {0: 0}
    res = -1  # 记录最短子数组长度，默认为 -1（不存在）

    # 遍历数组计算前缀和
    for i in range(len(nums)):
        n = nums[i]
        prefix_sum = sums[-1] + n  # 计算当前前缀和
        sums.append(prefix_sum)  # 存入前缀和列表
        
        # 查找是否存在 `prefix_sum - k`
        idx = hash_map.get(prefix_sum - k)
        if idx is not None:  # 说明找到了子数组和为 k
            length = i - idx + 1  # 计算当前子数组长度
            if res < 0 or length < res:  # 更新最短长度
                res = length
        
        # 存储当前前缀和对应的索引（若该前缀和未出现过）
        hash_map.setdefault(prefix_sum, i + 1)
    
    return res

```

---

## **逐步执行分析**

**输入**

`nums = [1, 1, 1, 2, 3, 4] k = 5`

---

### **初始化**

|变量|初始值|
|---|---|
|`sums = [0]`|记录前缀和|
|`hash_map = {0: 0}`|记录前缀和首次出现的索引|
|`res = -1`|最短子数组长度|

---

### **遍历 `nums`**

|`i`|`nums[i]`|`prefix_sum` 计算|`prefix_sum - k`|`hash_map` 查询|`res` 更新|`hash_map` 记录|
|---|---|---|---|---|---|---|
|0|1|`0 + 1 = 1`|`1 - 5 = -4`|不存在|-|`{0: 0, 1: 1}`|
|1|1|`1 + 1 = 2`|`2 - 5 = -3`|不存在|-|`{0: 0, 1: 1, 2: 2}`|
|2|1|`2 + 1 = 3`|`3 - 5 = -2`|不存在|-|`{0: 0, 1: 1, 2: 2, 3: 3}`|
|3|2|`3 + 2 = 5`|`5 - 5 = 0`|`hash_map[0] = 0`|`4 - 0 = 4`|`{0: 0, 1: 1, 2: 2, 3: 3, 5: 4}`|
|4|3|`5 + 3 = 8`|`8 - 5 = 3`|`hash_map[3] = 3`|`5 - 3 = 2` ✅|`{0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 8: 5}`|
|5|4|`8 + 4 = 12`|`12 - 5 = 7`|不存在|-|`{0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 8: 5, 12: 6}`|

最终返回：

`2`

最短子数组：

- `[2,3]`（长度=2）
- `[1,4]`（长度=2）

---

## **时间与空间复杂度分析**

1. **时间复杂度**
    
    - **`O(n)`**：遍历 `nums` 一次，每次 `O(1)` 计算 `prefix_sum` 和查询 `hash_map`，总复杂度是 `O(n)`。
2. **空间复杂度**
    
    - **`O(n)`**：`hash_map` 需要存储 `O(n)` 个前缀和。

---

## **其他解法**

### **1. 暴力法 `O(n^2)`**

- **思路**
    - 枚举所有子数组 `nums[i:j]`，计算 `sum(nums[i:j])` 并统计符合 `sum == k` 的最短子数组长度。
    - **缺点**：时间复杂度 `O(n^2)`，对于大数据不适用。
```python
def subarray_sum_equals_k_ii(nums, k):
    n = len(nums)
    res = float('inf')

    for i in range(n):
        total = 0
        for j in range(i, n):
            total += nums[j]
            if total == k:
                res = min(res, j - i + 1)

    return res if res != float('inf') else -1

```

---

## **方法比较**

|方法|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|
|**前缀和 + 哈希表（最佳解法）**|`O(n)`|`O(n)`|**大数据最优**|
|暴力法|`O(n^2)`|`O(1)`|**数据较小时可用**|

🚀 **前缀和 `O(n)` 解法是最优，适用于大规模数据！**