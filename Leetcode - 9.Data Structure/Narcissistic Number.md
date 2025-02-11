Lintcode 147
水仙花数的定义是，这个数等于他每一位数上的幂次之和 [见维基百科的定义](https://en.wikipedia.org/wiki/Narcissistic_number)

比如一个3位的十进制整数`153`就是一个水仙花数。因为 153 = 13 + 53 + 33。

而一个4位的十进制数`1634`也是一个水仙花数，因为 1634 = 14 + 64 + 34 + 44。

给出`n`，找到所有的n位十进制水仙花数。

**样例 1:**
```python
"""
输入: 1
输出: [0,1,2,3,4,5,6,7,8,9]
```
**样例 2:**
```python
"""
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