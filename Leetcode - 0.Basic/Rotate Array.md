Lintcode 8
给定一个字符数组 `s` 和一个偏移量offset，根据偏移量**原地旋转**字符数组(从左向右旋转)。

**样例 1：**
输入：
```python
s = "abcdefg"
offset = 3
```
输出：
```python
"efgabcd"
```
解释：
注意是**原地旋转**，即 s 旋转后为"efgabcd"

**样例 2：**
输入：
```python
s = "abcdefg"
offset = 0
```
输出：
```python
"abcdefg"
```
解释：
注意是**原地旋转**，即 s 旋转后为"abcdefg"

**样例 3：**
输入：
```python
s = "abcdefg"
offset = 1
```
输出：
```python
"gabcdef"
```
解释：
注意是**原地旋转**，即 s 旋转后为"gabcdef"

**样例 4：**
输入：
```python
s = "abcdefg"
offset = 2
```
输出：
```python
"fgabcde"
```
解释：
注意是**原地旋转**，即 s 旋转后为"fgabcde"

**样例 5：**
输入：
```python
s = "abcdefg"
offset = 10
```
输出：
```python
"efgabcd"
```
解释：
注意是**原地旋转**，即 s 旋转后为"efgabcd"


```python
def rotate_string(self, s: List[str], offset: int):
	if len(s) > 0:
		offset = offset % len(s)
		
	temp = (s + s)[len(s) - offset : 2 * len(s) - offset]

	for i in range(len(temp)):
		s[i] = temp[i]
```
pass