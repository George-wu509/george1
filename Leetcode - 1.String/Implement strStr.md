
**样例 1：**
输入：
```
source = "source"
target = "target"
```
输出：
```
-1        
```
解释：
如果source里没有包含target的内容，返回-1

**样例 2：**
输入：
```
source = "abcdabcdefg"
target = "bcd"
```
输出：
```
1             
```
解释：
如果source里包含target的内容，返回target在source里第一次出现的位置

**样例 3：**
输入：
```
source = "lintcode"
target = ""
```
输出：
```
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