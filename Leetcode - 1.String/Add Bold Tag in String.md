Lintcode 1127
给定一个字符串s和一个字符串列表dict，你需要添加一对封闭的粗体标记 `<b>` 和 `</b>` 来包装dict中存在的s中的子串。如果两个这样的子串重叠，则需要通过一对封闭的粗体标记将它们包装在一起。此外，如果由粗体标记包装的两个子字符串是连续的，则需要将它们组合在一起

```python
"""
输入: 
s = "abcxyz123"
dict = ["abc","123"]
输出:
"<b>abc</b>xyz<b>123</b>"
```

```python
"""
输入: 
s = "aaabbcc"
dict = ["aaa","aab","bc"]
输出:
"<b>aaabbc</b>c"
```


```python
def add_bold_tag(self, s: str, dict: List[str]) -> str:
	import itertools
	N = len(s)
	mask = [False] * N
	for i in range(N):
		prefix = s[i:]
		for word in dict:
			if prefix.startswith(word):
				for j in range(i, min(i+len(word), N)):
					mask[j] = True

	ans = []
	for incl, grp in itertools.groupby(zip(s, mask), lambda z: z[1]):
		if incl: ans.append("<b>")
		ans.append("".join(z[0] for z in grp))
		if incl: ans.append("</b>")
	return "".join(ans)
```
pass