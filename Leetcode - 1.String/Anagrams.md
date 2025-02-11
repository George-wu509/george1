Lintcode 171
给出一个字符串数组S，找到其中所有的乱序字符串(Anagram)。如果一个字符串是乱序字符串，那么他存在一个字母集合相同，但顺序不同的字符串也在S中。


```python
"""
输入:["lint", "intl", "inlt", "code"]
输出:["lint", "inlt", "intl"]
```

**样例 2:**

```python
"""
输入:["ab", "ba", "cd", "dc", "e"]
输出: ["ab", "ba", "cd", "dc"]
```


```python
def anagrams(self, strs):
	dict = {}
	for word in strs:
		sortedword = ''.join(sorted(word))
		dict[sortedword] = [word] if sortedword not in dict else dict[sortedword] + [word]
	res = []
	for item in dict:
		if len(dict[item]) >= 2:
			res += dict[item]
	return res
```
pass