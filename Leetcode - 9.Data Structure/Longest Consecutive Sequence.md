
**样例 1：**
输入：
```
num = [100, 4, 200, 1, 3, 2]
```
输出：
```
4
```
解释：
这个最长的连续序列是 [1, 2, 3, 4]. 返回所求长度 4


```python
def longest_consecutive(self, num: List[int]) -> int:
	longest_streak = 0
	num_set = set(num)

	for nu in num_set:
		if nu - 1 not in num_set:
			current_num = nu
			current_streak = 1

			while current_num + 1 in num_set:
				current_num += 1
				current_streak += 1

			longest_streak = max(longest_streak, current_streak)

	return longest_streak
```
pass