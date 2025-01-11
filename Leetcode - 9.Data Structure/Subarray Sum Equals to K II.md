
**样例1：**
```
输入: 
nums = [1,1,1,2] and k = 3
输出: 
2
```
**样例2：**
```
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