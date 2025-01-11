**样例 1:**
```
输入：[1,2,1,3,4,3,5,1,2]
输出：3
解释：
选择[1, 2, 1] 或 [3, 4, 3]。 长度是3。
```
**样例 2:**
```
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