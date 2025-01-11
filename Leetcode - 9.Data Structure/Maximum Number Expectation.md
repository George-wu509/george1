
**样例 1:**
```
输入: [1, 2, 3]
输出: 2.33
说明: 
一共有六个子区间，分别是 [1], [2], [3], [1, 2], [2, 3], [1, 2, 3]
分别的最大值是 1, 2, 3, 2, 3, 3
每一个最大值出现的概率是1 / 6，
所以最大值期望是 7 / 3,
得到答案: 2.33
```
**样例 2:**
```
输入: [2, 3, 2]
输出: 2.67
说明: 
一共有六个子区间，分别是 : [2], [3], [2], [2, 3], [3, 2], [2, 3, 2]
分别的最大值是 :2, 3, 2, 3, 3, 3
每一个最大值出现的概率是 1 / 6，
所以最大值期望是 8 / 3,
得到答案 : 2.67
```


```python
def expect_maximum(self, nums: List[int]) -> float:
	dp, stack = [0], []
	for i in range(len(nums)):
		while stack and nums[stack[-1]] <= nums[i]:
			stack.pop(-1)
		length = i - stack[-1] if stack else i + 1
		dp.append(length * nums[i] + dp[i + 1 - length])
		stack.append(i)
	count = len(nums) * (len(nums) + 1) // 2
	return sum(dp) / count
```
pass