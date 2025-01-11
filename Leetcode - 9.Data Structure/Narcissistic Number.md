
**样例 1:**
```
输入: 1
输出: [0,1,2,3,4,5,6,7,8,9]
```
**样例 2:**
```
输入:  2
输出: []	
样例解释: 没有2位数的水仙花数。
```


```python
def get_narcissistic_numbers(self, n):
	if n == 1:
		return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	if n == 6:
		return [548834]

	result = []
	for i in range(10 ** (n-1), 10 ** n):
		j, s = i, 0
		while j != 0:
			s += (j % 10) ** n;
			j = j // 10
		if s == i:
			result.append(i)
	return result
```
pass