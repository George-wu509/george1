
**样例 1:**
```
输入:[1, 2, 3]
输出:0
说明: 
这个数组有6个子区间: [1], [2], [3], [1, 2], [2, 3], [1, 2, 3]
分别对应的异或和是: 0, 0, 0, 3, 1, 2
最后的答案是: 0
```
**样例 2:**
```
输入:[1, 3, 2]
输出:1
说明: 
这个数组有6个子区间: [1], [3], [2], [1, 3], [3, 2], [1, 3, 2]
分别对应的异或和是: 0, 0, 0, 2, 1, 2
最后的答案是: 1
```


```python
def xor_sum(self, nums):
	n = len(nums)
	left_min, left_max = [0] * n, [0] * n
	right_min, right_max = [0] * n, [0] * n
	
	stack = []
	for i in range(n):
		while stack and nums[i] > nums[stack[-1]]:
			stack.pop(-1)
		left_min[i] = i - stack[-1] - 1 if stack else i
		stack.append(i)
	
	stack = []
	for i in range(n):
		while stack and nums[i] < nums[stack[-1]]:
			stack.pop(-1)
		left_max[i] = i - stack[-1] - 1 if stack else i
		stack.append(i)
	
	stack = []
	for i in range(n - 1, -1, -1):
		while stack and nums[i] >= nums[stack[-1]]:
			stack.pop(-1)
		right_min[i] = stack[-1] - i - 1 if stack else n - 1 - i
		stack.append(i)
	
	stack = []
	for i in range(n - 1, -1, -1):
		while stack and nums[i] <= nums[stack[-1]]:
			stack.pop(-1)
		right_max[i] = stack[-1] - i - 1 if stack else n - 1 - i
		stack.append(i)
	
	result = 0
	for i in range(n):
		times = left_min[i] + right_min[i] + 1
		times += left_min[i] * right_min[i]
		times += left_max[i] + right_max[i] + 1
		times += left_max[i] * right_max[i]
		result ^= (times % 2) * nums[i]
	
	return result
```
pass