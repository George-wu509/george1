Lintcode 209
给出一个字符串，找出第一个只出现一次的字符。假设只出现一次的字符数量大于等于1。


```python
"""
样例 1:
	输入: "abaccdeff"
	输出:  'b'
	
	解释:
	'b' 是第一个出现一次的字符


样例 2:
	输入: "aabccd"
	输出:  'b'
	
	解释:
	'b' 是第一个出现一次的字符
```


```python
from collections import Counter

def first_uniq_char(self, str):
	counter = Counter(str)

	for c in str:
		if counter[c] == 1:
			return c
```
pass