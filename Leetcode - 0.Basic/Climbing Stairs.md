
**样例 1：**
输入：
```
n = 3
```
输出：
```
3
```
解释：
1. 1, 1, 1
2. 1, 2
3. 2, 1
共3种

**样例 2：**
输入：
```
n = 1
```
输出：
```
1
```
解释：
只有一种方案


```python
def climb_stairs(self, n: int) -> int:
	if n == 0:
		return 0
	if n <= 2:
		return n
	result=[1,2]
	for i in range(n-2):
		result.append(result[-2]+result[-1])
	return result[-1]
```
pass
