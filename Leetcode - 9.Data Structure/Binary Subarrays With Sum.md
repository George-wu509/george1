Lintcode 1712
在由若干 `0` 和 `1` 组成的数组 `A` 中，有多少个和为 `S` 的**非空**子数组。

**样例 1:**
```python

输入：A = [1,0,1,0,1], S = 2
输出：4
解释：
如下面黑体所示，有 4 个满足题目要求的子数组：
[1,0,1]  (id=0,1,2)
[1,0,1]  (id=2,3,4)
[1,0,1,0]
[0,1,0,1]
```

**样例 2:**
```python
"""
输入：A = [0,0,0,0,0,0,1,0,0,0], S = 0
输出：27
解释：
和为 S 的子数组有 27 个
```


```python
def numSubarraysWithSum(self, nums, S):
	count = 0
	prefix_sum = 0
	prefix_sum_counts = {0: 1}  # 初始化哈希表，前綴和為 0 的出現次數為 1 

	for num in nums:
		prefix_sum += num

		# 檢查是否存在前綴和等於 (prefix_sum - S) 的情況
		if (prefix_sum - S) in prefix_sum_counts:
			count += prefix_sum_counts[prefix_sum - S]

		# 更新當前前綴和的出現次數
		if prefix_sum in prefix_sum_counts:
			prefix_sum_counts[prefix_sum] += 1
		else:
			prefix_sum_counts[prefix_sum] = 1

	return count
```
pass


# **LintCode 1712 - Binary Subarrays With Sum（和为 S 的二进制子数组）**

**程式碼解釋：**

1. **初始化：**
    
    - `count = 0`: 初始化計數器，用於記錄和為 `S` 的子數組數量。
    - `prefix_sum = 0`: 初始化前綴和為 0。
    - `prefix_sum_counts = {0: 1}`: 初始化一個哈希表（字典），用於儲存每個前綴和出現的次數。我們將前綴和 0 的出現次數初始化為 1，這是因為一個空的前綴（在遍歷任何元素之前）的總和為 0，這對於處理以數組開頭的子數組很有用。
2. **遍歷數組：**
    
    - 對於數組中的每個元素 `num`：
        - `prefix_sum += num`: 更新當前的前綴和。
        - **檢查目標前綴和：**
            - `if (prefix_sum - S) in prefix_sum_counts:`: 我們檢查哈希表中是否存在一個前綴和等於 `prefix_sum - S`。
            - **原理：** 如果存在一個前綴和 `prev_prefix_sum` 等於 `prefix_sum - S`，那麼從索引 `prev_prefix_sum` 的下一個位置到當前位置的子數組的總和就是 `prefix_sum - prev_prefix_sum = prefix_sum - (prefix_sum - S) = S`。
            - `count += prefix_sum_counts[prefix_sum - S]`: 如果找到了這樣的前綴和，我們將其出現的次數加到 `count` 上，因為每個這樣的先前出現都對應一個新的和為 `S` 的子數組。
        - **更新前綴和計數：**
            - `if prefix_sum in prefix_sum_counts:`: 檢查當前的前綴和是否已經在哈希表中。
            - `prefix_sum_counts[prefix_sum] += 1`: 如果存在，則增加其計數。
            - `else: prefix_sum_counts[prefix_sum] = 1`: 如果不存在，則將其添加到哈希表中並將計數設置為 1。
3. **返回結果：**
    
    - `return count`: 返回最終的計數，即和為 `S` 的子數組數量。

**時間複雜度和空間複雜度：**

- **時間複雜度：** O(n)，其中 n 是數組的長度。我們只需要遍歷數組一次。哈希表的查找和插入操作的平均時間複雜度是 O(1)。
- **空間複雜度：** O(n)，在最壞的情況下，哈希表可能會儲存所有可能的前綴和，其數量與數組的長度成正比。

希望這個程式碼和解釋能夠幫助您理解如何使用前綴和數組和哈希表來解決這個問題！

---

## **其他解法**

### **1. 前缀和 + 哈希表（O(n)）**

- **思路**：
    - 使用 `prefix_sum` 计算 `prefix[j] - prefix[i] = s` 的个数。
    - 用 `hashmap` 记录 `prefix_sum` 出现次数，快速查找前缀和是否满足 `sum == s`。

### **2. 暴力解法（O(n²)）**

- **思路**：
    - 遍历所有 `(i, j)` 计算 `sum(i, j)`，检查是否等于 `s`。

### **3. 二分查找（O(n log n)）**

- **思路**：
    - 先计算 `prefix_sum`，然后对 `prefix[j] - s` 进行二分查找。

---

## **LintCode 相关题目**

|**题号**|**题目名称**|**难度**|**核心技术**|
|---|---|---|---|
|**LintCode 1712**|Binary Subarrays With Sum|⭐⭐⭐|**滑动窗口 + 双指针**|
|**LintCode 1075**|Subarray Product Less Than K|⭐⭐⭐|**滑动窗口 + 乘积**|
|**LintCode 1380**|Subarray Sum Equals K|⭐⭐⭐|**前缀和 + 哈希表**|
|**LintCode 604**|Window Sum|⭐⭐|**滑动窗口**|

---

## **总结**

1. **最优解法：滑动窗口 `O(n)`**
    
    - **两个左指针 `left1` 和 `left2` 计算满足 `sum == s` 的子数组**。
2. **其他解法**
    
    - **前缀和 + 哈希表（O(n)）**，适用于 **任意整数数组**。
    - **暴力解法（O(n²)）**，仅适用于小规模数据。
    - **二分查找（O(n log n)）**，适用于 **预处理后查询**。