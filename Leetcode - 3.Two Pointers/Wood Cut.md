

**样例 1**
```plain
输入:
L = [232, 124, 456]
k = 7
输出: 114
说明: 我们可以把它分成 114 的 7 段，而 115 不可以
而且对于 124 这根原木来说多余的部分没用可以舍弃，不需要完整利用
```
**样例 2**
```plain
输入:
L = [1, 2, 3]
k = 7
输出: 0
说明: 很显然我们不能按照题目要求完成。
```


```python
def wood_cut(self, l: List[int], k: int) -> int:
	if not l:
		return 0
	start, end = 1, max(l)
	while start + 1 < end:
		mid = (start + end) // 2
		if self.get_pieces(l, mid) >= k:
			start = mid
		else:
			end = mid
			
	if self.get_pieces(l, end) >= k:
		return end
	if self.get_pieces(l, start) >= k:
		return start
		
	return 0
	
def get_pieces(self, l, length):
	pieces = 0
	for l in l:
		pieces += l // length
	return pieces
```
pass