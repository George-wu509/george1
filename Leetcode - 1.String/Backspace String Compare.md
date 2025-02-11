Lintcode 1425
给定 `S` 和 `T` 两个字符串，当它们分别被输入到空白的文本编辑器后，判断二者是否相等，并返回结果。 `#` 代表退格字符。

**样例 1：**
```python
"""
输入：S = "ab#c", T = "ad#c"
输出：true
解释：S 和 T 都会变成 “ac”。
```
**样例 2：**
```python
"""
输入：S = "ab##", T = "c#d#"
输出：true
解释：S 和 T 都会变成 “”。
```
**样例 3：**
```python
"""
输入：S = "a##c", T = "#a#c"
输出：true
解释：S 和 T 都会变成 “c”。
```
**示例 4：**
```python
"""
输入：S = "a#c", T = "b"
输出：false
解释：S 会变成 “c”，但 T 仍然是 “b”。
```


```python
def backspace_compare(self, s: str, t: str) -> bool:
	i, j = len(s) - 1, len(t) - 1
	skipS = skipT = 0

	while i >= 0 or j >= 0:
		while i >= 0:
			if s[i] == "#":
				skipS += 1
				i -= 1
			elif skipS > 0:
				skipS -= 1
				i -= 1
			else:
				break
		while j >= 0:
			if t[j] == "#":
				skipT += 1
				j -= 1
			elif skipT > 0:
				skipT -= 1
				j -= 1
			else:
				break
		if i >= 0 and j >= 0:
			if s[i] != t[j]:
				return False
		elif i >= 0 or j >= 0:
			return False
		i -= 1
		j -= 1

	return True
```
pass