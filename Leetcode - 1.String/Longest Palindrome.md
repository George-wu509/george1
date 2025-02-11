Lintcode 627
给出一个包含大小写字母的字符串。求出由这些字母构成的最长的回文串的长度是多少。

数据是大小写敏感的，也就是说，`"Aa"` 并不会被认为是一个回文串。

```python
"""
输入 : s = "abccccdd"
输出 : 7
说明 : 
一种可以构建出来的最长回文串方案是 "dccaccd"。
```


```python
    def longest_palindrome(self, s):
        # Write your code here
        hash = {}

        for c in s:
            if c in hash:
                del hash[c]
            else:
                hash[c] = True

        remove = len(hash)
        if remove > 0:
            remove -= 1
    
        return len(s) - remove
```
pass