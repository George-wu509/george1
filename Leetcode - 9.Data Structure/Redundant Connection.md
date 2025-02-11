Lintcode 684
给出两个字符串，你需要找到缺少的字符串

**样例 1:**

```python
"""
输入 : str1 = "This is an example", str2 = "is example"
输出 : ["This", "an"]
```


```python
def missing_string(self, str1, str2):
	result = []
	dict = set(str2.split())
	
	for word in str1.split():
		if word not in dict:
			result.append(word)
			
	return result
```
pass