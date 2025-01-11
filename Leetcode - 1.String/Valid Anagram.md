
**样例 1:**
```
输入: s = "anagram", t = "nagaram"
输出: true
```
**样例 2:**
```
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