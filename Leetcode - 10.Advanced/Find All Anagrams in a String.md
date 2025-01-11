
**样例 1:**

```
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