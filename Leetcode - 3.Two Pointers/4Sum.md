Example
**样例 1：**
输入：
```
numbers = [2,7,11,15]
target = 3
```
输出：
```
[]
```
解释：
2 + 7 + 11 + 15 ！= 3，不存在满足条件的四元组。  

**样例 2：**
输入：
```
numbers = [1,0,-1,0,-2,2]
target = 0
```
输出：
```
[[-1, 0, 0, 1],[-2, -1, 1, 2],[-2, 0, 0, 2]]
```
解释：
有3个不同的四元组满足四个数之和为0。


```python
def four_sum(self, nums, target):
	nums.sort()
	res = []
	length = len(nums)
	for i in range(0, length - 3):
		if i and nums[i] == nums[i - 1]:
			continue
		for j in range(i + 1, length - 2):
			if j != i + 1 and nums[j] == nums[j - 1]:
				continue
			sum = target - nums[i] - nums[j]
			left, right = j + 1, length - 1
			while left < right:
				if nums[left] + nums[right] == sum:
					res.append([nums[i], nums[j], nums[left], nums[right]])
					right -= 1
					left += 1
					while left < right and nums[left] == nums[left - 1]:
						left += 1
					while left < right and nums[right] == nums[right + 1]:
						right -= 1
				elif nums[left] + nums[right] > sum:
					right -= 1
				else:
					left += 1
	return res
```
pass