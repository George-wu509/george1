Lintcode 13
对于一个给定的 `source` 字符串和一个 `target` 字符串，你应该在 source 字符串中找出 target 字符串出现的第一个位置(从`0`开始)。如果不存在，则返回 `-1`。

**样例 1：**
输入：
```python
"""
source = "source"
target = "target"
```
输出：
```python
"""
-1        
```
解释：
如果source里没有包含target的内容，返回-1

**样例 2：**
输入：
```python
"""
source = "abcdabcdefg"
target = "bcd"
```
输出：
```python
"""
1             
```
解释：
如果source里包含target的内容，返回target在source里第一次出现的位置

**样例 3：**
输入：
```python
"""
source = "lintcode"
target = ""
```
输出：
```python
"""```
0             
```


```python
def str_str(self, source: str, target: str) -> int:
	if source is None or target is None:
		return -1
	len_s = len(source)
	len_t = len(target)
	for i in range(len_s - len_t + 1):
		j = 0
		while (j < len_t):
			if source[i + j] != target[j]:
				break
			j += 1
		if j == len_t:
			return i
	return -1
```
pass