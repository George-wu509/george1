
**样例 1:**
```
输入: [0,1]
输出: 2
解释: [0, 1] 是具有相等数量的 0 和 1 的最长子数组。
```
**样例 2:**
```
输入: [0,1,0]
输出: 2
解释: [0, 1] (或者 [1, 0]) 是具有相等数量 0 和 1 的最长子数组。
```


```python
def find_max_length(self, nums: List[int]) -> int:
	map = {0:-1}
	flag = 0
	ans = 0
	for index, num in enumerate(nums):
		if num == 0:
			flag -= 1
		else:
			flag += 1
		if flag in map:
			ans = max(ans, index - map[flag])
		else:
			map[flag] = index
	return ans
```
pass