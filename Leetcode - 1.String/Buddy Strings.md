Lintcode 1510
给定两个由小写字母构成的字符串 `A` 和 `B`，如果能通过恰好交换一次 `A` 中的两个下标不同的字母后得到 `B`，就返回 `true`；否则返回 `false`。

**示例 1:**
```python
"""```
输入: A = "ab", B = "ba"
输出: true
```
**示例 2:**
```python
"""
输入: A = "ab", B = "ab"
输出: false
```
**示例 3:**
```python
"""
输入: A = "aa", B = "aa"
输出: true
```
**示例 4:**
```python
"""
输入: A = "aaaaaaabc", B = "aaaaaaacb"
输出: true
```
**示例 5:**
```python
"""
输入: A = "", B = "aa"
输出: false
```


```python
def buddyStrings(self, a: str, b: str) -> bool:
	if len(a) != len(b):
		return False
	if a == b:
		if len(set(a)) < len(b): 
			return True
		else:
			return False
	diff = [(aa, bb) for aa, bb in zip(a, b) if aa != bb]
	return len(diff) == 2 and diff[0][0] == diff[1][1] and diff[0][1] == diff[1][0]
```
pass