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
		for index, number in enumerate(numbers)
	]
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