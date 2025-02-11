Lintcode 1275
你的任务是计算 a^b mod 1337，其中 a 是一个正整数，b 是一个超级大的正整数，以数组的形式给出。

**样例1**
```python
"""
输入：
a = 2
b = [3]
输出：
8
```
**样例2**
```python
"""
输入：
a = 2
b = [1,0]
输出：
1024
```


```python
def super_pow(self, a: int, b: List[int]) -> int:
	MOD = 1337
	ans = 1
	for e in b:
		ans = pow(ans, 10, MOD) * pow(a, e, MOD) % MOD
	return ans
```
pass