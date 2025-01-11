
```
输入 : [1, 0, 1, 2]
输出 : [0, 1, 1, 2]
解释 : 原地排序。
```


```python
def sort_colors(self, nums):
	left, index, right = 0, 0, len(nums) - 1
	# be careful, index < right is not correct
	while index <= right:
		if nums[index] == 0:
			nums[left], nums[index] = nums[index], nums[left]
			left += 1
			index += 1 # move to next number
		elif nums[index] == 2:
			nums[right], nums[index] = nums[index], nums[right]
			right -= 1
		else:  # == 1, skip
			index += 1
```
pass