
**样例1**
```
输入: nums = [1,1,1] 和 k = 2
输出: 2
解释:
子数组 [0,1] 和 [1,2]
```
**样例2**
```
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