
**样例1**
```
输入：
a = 2
b = [3]
输出：
8
```
**样例2**
```
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