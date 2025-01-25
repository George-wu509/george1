1643
小红去果园采水果。有2个篮子，可以装无数个水果，但是只能装一种水果。从任意位置的树开始，往右采。遇到2种情况退出，1. 遇到第三种水果，没有篮子可以放了，2. 到头了。返回可以采摘的最多的水果个数。水果数组用`arr`表示。

**样例 1:**
```python
输入：[1,2,1,3,4,3,5,1,2]
输出：3
解释：
选择[1, 2, 1] 或 [3, 4, 3]。 长度是3。
```
**样例 2:**
```python
输入：[1,2,1,2,1,2,1]
输出：7
解释：
选择 [1, 2, 1, 2, 1, 2, 1]。长度是 7。
```



```python
def pick_fruits(self, arr: List[int]) -> int:
	ans = i = 0
	count = collections.Counter()
	for j, x in enumerate(arr):
		count[x] += 1
		while len(count) >= 3:
			count[arr[i]] -= 1
			if count[arr[i]] == 0:
				del count[arr[i]]
			i += 1
		ans = max(ans, j - i + 1)
	return ans
```
pass