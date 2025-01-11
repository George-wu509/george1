
**样例1:**
```
输入: 
nums = [1, 7, 3, 6, 5, 6]
输出: 3
解释: 
索引3 (nums[3] = 6)左侧所有数之和等于右侧之和。
并且3是满足条件的第一个索引。
```
**样例2:**
```
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
		if _sum * 2 + n == total: return i
		_sum += n
	return -1
```
pass