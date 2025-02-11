Lintcode


**样例 1:**
```python
"""
输入：S="jmeqksfrsdcmsiwvaovztaqenprpvnbstl"，T="u"
输出：""
解释： 无法匹配
```
**样例 2:**
```python
"""
输入：S = "abcdebdde"， T = "bde"
输出："bcde"
解释："bcde"是答案，"deb"不是一个较小的窗口，因为窗口中的T元素必须按顺序发生。
```



```python
class Solution:
    def min_window(self, s: str, t: str) -> str:
        cur = [i if x == t[0] else None
               for i, x in enumerate(s)]
        #At time j when considering t[:j+1],
        #the smallest window [s, e] where s[e] == t[j]
        #is represented by cur[e] = s.
        for j in range(1, len(t)):
            last = None
            new = [None] * len(s)
            #Now we would like to calculate the candidate windows
            #"new" for t[:j+1].  'last' is the last window seen.
            for i, u in enumerate(s):
                if last is not None and u == t[j]: new[i] = last
                if cur[i] is not None: last = cur[i]
            cur = new

        #Looking at the window data cur, choose the smallest length
        #window [s, e].
        ans = 0, len(s)
        for e, st in enumerate(cur):
            if st and st >= 0 and e - st < ans[1] - ans[0]:
                ans = st, e
        return s[ans[0]: ans[1]+1] if s and ans[1] < len(s) else ""
```
pass