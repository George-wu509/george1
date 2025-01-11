
```
输入:["lint", "intl", "inlt", "code"]
输出:["lint", "inlt", "intl"]
```

**样例 2:**

```
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