
例1:
```
输入: [1,2,1]
输出: [2,-1,2]
解释：第一个1的下一个更大的数字是2;
数字2找不到下一个更大的数字;
第二个1的下一个更大的数字需要循环搜索，答案也是2。
```
例2:
```
输入: [1]
输出: [-1]
解释：
数字1找不到下一个更大的数字
```


```python
def next_greater_elements(self, nums: List[int]) -> List[int]:
	n = len(nums)
	ret = [-1] * n
	stk = list()

	for i in range(n * 2 - 1):
		while stk and nums[stk[-1]] < nums[i % n]:
			ret[stk.pop()] = nums[i % n]
		stk.append(i % n)
	
	return ret
```
pass