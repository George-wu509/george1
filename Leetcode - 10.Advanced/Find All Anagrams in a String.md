Lintcode 647
给定一个字符串 `s` 和一个 **非空字符串** `p` ，找到在 `s` 中所有关于 `p` 的字谜的起始索引。  
如果`s`是`p`的一个字谜，则`s`是`p`的一个排列。  
字符串仅由小写英文字母组成，字符串 **s** 和 **p** 的长度不得大于 40,000。  
输出顺序无关紧要。

**样例 1:**

```python
"""
输入 : s = "cbaebabacd", p = "abc"
输出 : [0, 6]
说明 : 
子串起始索引 index = 0 是 "cba"，是"abc"的字谜。
子串起始索引 index = 6 是 "bac"，是"abc"的字谜。
```



```python
    def find_anagrams(self, s, p):
        list = []
        times = dict()
        # 存储p中每个字母出现的次数
        for c in p:
            if c not in times:
                times[c] = 1
            else:
                times[c] += 1
        l,r = 0,-1
        # s[l...r]为滑动窗口
        while l < len(s):
            if r - l + 1 == len(p):
                list.append(l)
            if r + 1 < len(s) and s[r + 1] in times and times[s[r + 1]] > 0:
                r += 1
                times[s[r]] -= 1
            else :
                if s[l] not in times:
                    times[s[l]] = 1
                else:
                    times[s[l]] += 1
                l += 1;
        return list
```
pass