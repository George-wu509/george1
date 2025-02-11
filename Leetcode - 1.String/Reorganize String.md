Lintcode 1041
给定一个字符串`S`，检查是否能重新排布其中的字母，使得两相邻的字符不同。

若可行，输出任意可行的结果。若不可行，返回空字符串。

**样例 1:**
```python
"""
输入: S = "aab"
输出: "aba"
```
**样例 2:**
```python
"""
输入: S = "aaab"
输出: ""
```


```python
class Solution:
    def reorganize_string(self, s: str) -> str:
        if len(s) < 2:
            return s

        length = len(s)
        counts = collections.Counter(s)
        maxCount = max(counts.items(), key=lambda x: x[1])[1]
        if maxCount > (length + 1) // 2:
            return ""

        reorganizeArray = [""] * length
        evenIndex, oddIndex = 0, 1
        halfLength = length // 2

        for c, count in counts.items():
            while count > 0 and count <= halfLength and oddIndex < length:
                reorganizeArray[oddIndex] = c
                count -= 1
                oddIndex += 2
            while count > 0:
                reorganizeArray[evenIndex] = c
                count -= 1
                evenIndex += 2

        return "".join(reorganizeArray)
```
pass