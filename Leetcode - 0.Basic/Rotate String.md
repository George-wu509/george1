
**样例 1：**
输入：
```
s = "abcdefg"
offset = 3
```
输出：
```
"efgabcd"
```
解释
注意是**原地旋转**，即 s 旋转后为"efgabcd"

**样例 2：**
输入：
```
s = "abcdefg"
offset = 0
```
输出：
```
"abcdefg"
```
解释：
注意是**原地旋转**，即 s 旋转后为"abcdefg"

**样例 3：**
输入：
```
s = "abcdefg"
offset = 1
```
输出：
```
"gabcdef"
```
解释：
注意是**原地旋转**，即 s 旋转后为"gabcdef"

**样例 4：**
输入：
```
s = "abcdefg"
offset = 2
```
输出：
```
"fgabcde"
```
解释：
注意是**原地旋转**，即 s 旋转后为"fgabcde"

**样例 5：**
输入：
```
s = "abcdefg"
offset = 10
```
输出：
```
"efgabcd"
```
解释：
注意是**原地旋转**，即 s 旋转后为"efgabcd"



```python
class Solution:
    # @param s: a list of char
    # @param offset: an integer 
    # @return: nothing
    def rotateString(self, s, offset):
        if len(s) > 0:
            offset = offset % len(s)
            
        temp = (s + s)[len(s) - offset : 2 * len(s) - offset]

        for i in range(len(temp)):
            s[i] = temp[i]
```
pass