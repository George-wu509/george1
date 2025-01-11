
```
输入: 
s = "abcxyz123"
dict = ["abc","123"]
输出:
"<b>abc</b>xyz<b>123</b>"
```

```
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