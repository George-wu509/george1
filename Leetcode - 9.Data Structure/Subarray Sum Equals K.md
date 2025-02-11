Lintcode 838
给定一个整数数组和一个整数k，你需要找到连续子数列的和为k的总个数。

**样例1**
```python
"""
输入: nums = [1,1,1] 和 k = 2
输出: 2
解释:
子数组 [0,1] 和 [1,2]
```
**样例2**
```python
"""
输入: nums = [2,1,-1,1,2] 和 k = 3
输出: 4
解释:
子数组 [0,1], [1,4], [0,3] and [3,4]
```


```python
def subarray_sum_equals_k(self, nums: List[int], k: int) -> int:
	count, pre = 0, 0
	mp = {}
	mp[0] = 1
	for i in range(len(nums)):
		pre += nums[i]
		if (pre - k) in mp:
			count += mp.get(pre - k)
		if pre not in mp:
			mp[pre] = 0
		mp[pre] += 1
	return count
```
pass

解釋:
step1


# **LintCode 838: Subarray Sum Equals K（子数组和等于 K）**

---

## **题目描述**

给定一个整数数组 `nums` 和一个目标值 `k`，找到 **连续子数组的个数**，使得其元素之和等于 `k`。
#### **示例**

`输入： nums = [1, 1, 1] k = 2  输出： 2`

**解释**
```python
满足条件的子数组有：
[1, 1]（索引 0 到 1）
[1, 1]（索引 1 到 2）
```
---

## **解法：前缀和 + 哈希表（双指针）**

### **核心思路**

1. **利用前缀和**
    
    - 计算 `prefix_sum[i]`，表示 `nums[0]` 到 `nums[i]` 的累积和： 
    - 若 `prefix_sum[j] - prefix_sum[i] = k`，说明从 `i+1` 到 `j` 的子数组和为 `k`： 
    
2. **哈希表存储前缀和**
    - 维护 `mp`（哈希表）：记录**某个前缀和出现的次数**。
    - **如果 `prefix_sum - k` 在 `mp` 中，说明存在某个子数组满足 `sum == k`**，计数增加 `mp[prefix_sum - k]`。

---

## **代码解析**
```python
def subarray_sum_equals_k(self, nums: List[int], k: int) -> int:
    count, pre = 0, 0  # count 记录符合条件的子数组个数，pre 记录当前前缀和
    mp = {}  # 哈希表存储前缀和出现次数
    mp[0] = 1  # 处理从索引 0 开始的子数组

    for i in range(len(nums)):
        pre += nums[i]  # 计算当前前缀和

        if (pre - k) in mp:  # 查找是否存在 prefix_sum - k
            count += mp.get(pre - k)  # 增加符合条件的子数组个数

        if pre not in mp:
            mp[pre] = 0  # 初始化前缀和计数
        mp[pre] += 1  # 记录前缀和出现的次数

    return count

```

---

## **逐步执行分析**

**输入**
`nums = [1, 1, 1] k = 2`

---

### **初始化**

|变量|初始值|
|---|---|
|`count = 0`|记录符合条件的子数组数量|
|`pre = 0`|当前前缀和|
|`mp = {0: 1}`|记录前缀和 `0` 出现 1 次|

---

### **遍历 `nums`**

|`i`|`nums[i]`|`pre += nums[i]`|`pre - k`|`mp[pre - k]` 查询|`count` 更新|`mp` 记录|
|---|---|---|---|---|---|---|
|0|1|`1`|`1 - 2 = -1`|不在 `mp` 中|`0`|`mp = {0: 1, 1: 1}`|
|1|1|`2`|`2 - 2 = 0`|`mp[0] = 1`|`1`|`mp = {0: 1, 1: 1, 2: 1}`|
|2|1|`3`|`3 - 2 = 1`|`mp[1] = 1`|`2`|`mp = {0: 1, 1: 1, 2: 1, 3: 1}`|

最终返回：

`2`

**满足 `sum == k` 的子数组**

- `[1, 1]`（索引 `0~1`）
- `[1, 1]`（索引 `1~2`）

---

## **时间与空间复杂度分析**

3. **时间复杂度**
    
    - **`O(n)`**：遍历 `nums` 一次，每次 `O(1)` 计算 `pre` 和查询 `mp`，总复杂度是 `O(n)`。
4. **空间复杂度**
    
    - **`O(n)`**：最坏情况下，`mp` 需要存储 `O(n)` 个前缀和。

---

## **其他解法**

### **1. 暴力法 `O(n^2)`**

- **思路**
    - 枚举所有子数组 `nums[i:j]`，计算和 `sum(nums[i:j])` 并统计符合 `sum == k` 的子数组个数。
    - **缺点**：时间复杂度 `O(n^2)`，对于大数据不适用。
    
```python
def subarray_sum_equals_k(nums, k):
    count = 0
    for i in range(len(nums)):
        total = 0
        for j in range(i, len(nums)):
            total += nums[j]
            if total == k:
                count += 1
    return count
```

---

## **方法比较**

|方法|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|
|**前缀和 + 哈希表（最佳解法）**|`O(n)`|`O(n)`|**大数据最优**|
|暴力法|`O(n^2)`|`O(1)`|**数据较小时可用**|

🚀 **前缀和 `O(n)` 解法是最优，适用于大规模数据！**