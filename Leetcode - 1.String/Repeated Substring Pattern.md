
**样例1：**
```.
输入："abab"

输出：True

说明：可以由它的子串"ab"重复两次得到。
```
**样例2：**
```.
输入："aba"

输出：False
```
**样例3：**
```.
输入："abcabcabcabc"

输出：True

说明：可以由它的子串"abc"重复四次得到（同时也可以是"abcabc"重复两次）。
```


```python
    def repeated_substring_pattern(self, s: str) -> bool:
        l = len(s)
        next = [-1 for i in range(l)]
        j = -1
        for i in range(1, l):
            while j >= 0 and s[i] != s[j + 1]:
                j = next[j]
            if s[i] == s[j + 1]:
                j += 1
            next[i] = j
        lenSub = l - 1 - next[l - 1]
        return lenSub != l and l % lenSub ==0
```
pass