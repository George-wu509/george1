
**样例 1：**
```
输入：string[] = "Mr John Smith" and length = 13
输出：string[] = "Mr%20John%20Smith" and return 17
解释：
对于字符串 "Mr John Smith"，长度为 13。替换空格之后，参数中的字符串需要变为 "Mr%20John%20Smith"，并且把新长度 17 作为结果返回。
```
**样例 2：**
```
输入：string[] = "LintCode and Jiuzhang" and length = 21
输出：string[] = "LintCode%20and%20Jiuzhang" and return 25
解释：
对于字符串 "LintCode and Jiuzhang"，长度为 21。替换空格之后，参数中的字符串需要变为 "LintCode%20and%20Jiuzhang"，并且把新长度 25 作为结果返回
```



```python
def replaceBlank(self, string, length):
	if string is None:
		return length
		
	spaces = 0
	for c in string:
		if c == ' ':
			spaces += 1
	
	L = length + spaces * 2
	index = L - 1
	for i in range(length - 1, -1, -1):
		if string[i] == ' ':
			string[index] = '0'
			string[index - 1] = '2'
			string[index - 2] = '%'
			index -= 3
		else:
			string[index] = string[i]
			index -= 1
	return L
```
pass