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

        #如果字符串s 不等于 t, 但是他们生成的字典（{字符:计数}）相同，则为true
        if s == t:
            return False
        dict_1 = {}
        dict_2 = {}
        for ch in s:
            if ch in dict_1:
                dict_1[ch] += 1
            else:
                dict_1[ch] = 1

        for ch in t:
            if ch in dict_2:
                dict_2[ch] += 1
            else:
                dict_2[ch] = 1     

        if dict_1 == dict_2:
            return True
        return False
```
pass