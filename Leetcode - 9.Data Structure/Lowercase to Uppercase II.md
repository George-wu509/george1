
**样例 1:**
```
输入: str = "abc"
输出: "ABC"
```
**样例 2:**
```
输入: str = "aBc"
输出: "ABC"
```
**样例 3:**
```
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