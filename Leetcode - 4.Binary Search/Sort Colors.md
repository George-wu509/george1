Lintcode 148
给定一个包含红，白，蓝且长度为 n 的数组，将数组元素进行分类使相同颜色的元素相邻，并按照红、白、蓝的顺序进行排序。

我们使用整数 0，1 和 2 分别代表红，白，蓝

```python
"""
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