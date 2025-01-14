
例1:
```
输入: nums = [1, 2, 4, 8, 6, 3] 
输出: 8
```
例2:
```
输入: nums = [10, 9, 8, 7], 
输出: 10
```


```python
class Solution:
	def mountain_sequence(self, nums):    #My version - leetcode pass
		if not nums:
			return -1
		start, end = 0, len(nums)-1

		while start+1 < end:
			mid = start +(end-start)//2
			if nums[mid] < nums[mid-1]:
				end = mid
			elif nums[mid] < nums[mid+1]:
				start = mid
			else:
				return nums[mid]

		if nums[start] >= nums[end]:
			return nums[start]
		else:
			return nums[end]
```
pass