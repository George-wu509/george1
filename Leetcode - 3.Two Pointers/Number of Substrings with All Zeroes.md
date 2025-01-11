
**样例1:**
```
输入:"00010011"
输出:9
解释:
"0"子字符串有5个,
"00"子字符串有3个,
"000"子字符串有1个。
所以返回9
```
**样例2:**
```
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