
**样例 1：**
```
输入：
"MPMPCPMCMDEFEGDEHINHKLIN"
输出：
[9,7,8]
解释：
"MPMPCPMCM"
"DEFEGDE"
"HINHKLIN"
```


```python
def split_String(self, s):
	n = len(s)
	st = ed = -1
	last = [0 for _ in range(26)]
	ans = []
	for i in range(n):
		last[ord(s[i]) - 65] = i
	for i in range(n):
		if ed == -1:
			st = i
			ed = last[ord(s[i]) - 65]
			if st == ed:
				ans.append(1)
				ed = -1
		elif i == ed:
			ans.append(ed - st + 1)
			ed = -1
		else:
			ed = max(ed, last[ord(s[i]) - 65])
	return ans

```
pass