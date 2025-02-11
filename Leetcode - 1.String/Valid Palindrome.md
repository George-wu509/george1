Lintcode 415
给定一个字符串，判断其是否为一个回文串。只考虑字母和数字，并忽略大小写。
**样例 1:**
```python
"""
输入: "A man, a plan, a canal: Panama"
输出: true
解释: "amanaplanacanalpanama"
```
**样例 2:**
```python
"""
输入: "race a car"
输出: false
解释: "raceacar"
```
**样例 3:**
```python
"""
输入: "1b , 1"
输出: true
解释: "1b1"
```


```python
    def isPalindrome(self, s):
        start, end = 0, len(s) - 1
        while start < end:
            while start < end and not s[start].isalpha() and not s[start].isdigit():
                start += 1
            while start < end and not s[end].isalpha() and not s[end].isdigit():
                end -= 1
            if start < end and s[start].lower() != s[end].lower():
                return False
            start += 1
            end -= 1
        return True
```
pass