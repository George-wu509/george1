

**样例 1:**
```
输入：A = [1], K = 1
输出：1
```
**样例 2:**
```
输入：A = [1,2], K = 4
输出：-1
```



```python
from collections import deque
def shortest_subarray(self, a, k):
	mono_queue = deque([(0, -1)])
	shortest = float('inf')
	prefix_sum = 0
	for end in range(len(a)):
		prefix_sum += a[end]
		# pop left
		while mono_queue and prefix_sum - mono_queue[0][0] >= k:
			min_prefix_sum, index = mono_queue.popleft()
			shortest = min(shortest, end - index)
		# push right
		while mono_queue and mono_queue[-1][0] >= prefix_sum:
			mono_queue.pop()
		mono_queue.append((prefix_sum, end))
	
	if shortest == float('inf'):
		return -1
	return shortest
```
pass