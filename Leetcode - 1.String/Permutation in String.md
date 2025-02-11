Lintcode 1169
给定两个字符串`s1`和`s2`，如果`s2`包含`s1`的排列，则写一个函数返回true。 换句话说，第一个字符串的排列之一是第二个字符串的`substring`

**样例1:**
```python
"""
输入: s1 = "ab" s2 = "eidbaooo"
输出: true
解释: s2包含s1的一个排列("ba").
```
**样例2:**
```python
"""
输入: s1= "ab" s2 = "eidboaoo"
输出: false
```


```python
    def check_inclusion(self, s1: str, s2: str) -> bool:
        n,m = len(s1), len(s2)
        if n > m: return False
        cnt1, cnt2 = [0]*26, [0]*26
        for i in range(n):
            cnt1[ord(s1[i])-ord('a')] += 1
            cnt2[ord(s2[i])-ord('a')] += 1
        if cnt1 == cnt2:
            return True
        for i in range(n,m):
            cnt2[ord(s2[i])-ord('a')] += 1
            cnt2[ord(s2[i-n])-ord('a')] -= 1
            if cnt1 == cnt2:
                return True
        return False
```
pass