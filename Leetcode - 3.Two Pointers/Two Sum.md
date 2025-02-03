56
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

排序的兩種方法
nums = **sorted**(nums)
nums.**sort**()