1870
给出一个只包含`0`或`1`的字符串`str`,请返回这个字符串中全为`0`的子字符串的个数

**样例1:**
```python
输入:"00010011"
输出:9
解释:
"0"子字符串有5个,
"00"子字符串有3个,
"000"子字符串有1个。
所以返回9
```
**样例2:**
```python
输入:
"010010"
输出:
5
```


```python
def string_Count(self, str):       
	answer = 0
	count = 0
	for i in str:
		if i == '0':
			count += 1
		else:
			answer += (1 + count) * count // 2;
			count = 0       
	if count != 0:
		answer += (1 + count) * count // 2;       
	return answer
```
pass