Lintcode 773
给定两个字符串 _s_ 和 _t_ ，编写一个函数来判断 _t_ 是否是 _s_ 的字母异位词。

**样例 1:**
```python
"""
输入: s = "anagram", t = "nagaram"
输出: true
```
**样例 2:**
```python
"""
输入: s = "rat", t = "car"
输出: false
```



```python
    def is_anagram(self, s, t):
        if len(s) != len(t):
            return False
        return sorted(s) == sorted(t)
```
pass