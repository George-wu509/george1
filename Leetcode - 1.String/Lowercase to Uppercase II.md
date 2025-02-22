Lintcode 146
将一个字符串中的小写字母转换为大写字母。不是字母的字符不需要做改变。

**样例 1:**
```python
"""
输入: str = "abc"
输出: "ABC"
```
**样例 2:**
```python
"""
输入: str = "aBc"
输出: "ABC"
```
**样例 3:**
```python
"""
输入: str = "abC12"
输出: "ABC12"
```


```python
    def lowercase_to_uppercase2(self, letters):
        p = list(letters)
        #遍历整个字符串，将所有的小写字母转成大写字母
        for i in range(len(p)):
            if p[i] >= 'a' and p[i] <= 'z':
                p[i] = chr(ord(p[i]) - 32)
        return ''.join(p)
```
pass