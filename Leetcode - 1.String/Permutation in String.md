Lintcode 1169
给定两个字符串`s1`和`s2`，如果`s2`包含`s1`的排列，则写一个函数返回true。 换句话说，第一个字符串的排列之一是第二个字符串的`substring`

**样例1:**
```python
输入: s1 = "ab" s2 = "eidbaooo"
输出: true
解释: s2包含s1的一个排列("ba").
```
**样例2:**
```python
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
解釋:
step1: s1的長度為n, 先求得s1的字元頻率(可用collections.Counter).
step2: 在s2上遍例長度為n的子字串, 計算字元頻率, 並和s1的字元頻率比較


#### 滑动窗口

由于排列不会改变字符串中每个字符的个数，所以只有当两个字符串每个字符的个数均相等时，一个字符串才是另一个字符串的排列。

根据这一性质，记 s1​ 的长度为 n，我们可以遍历 s2 中的每个长度为 n 的子串，判断子串和 s1​ 中每个字符的个数是否相等，若相等则说明该子串是 s1​ 的一个排列。

使用两个数组 cnt1和 cnt2​，cnt1​ 统计 s1​ 中各个字符的个数，cnt2​ 统计当前遍历的子串中各个字符的个数。

由于需要遍历的子串长度均为 n，我们可以使用一个固定长度为 n 的**滑动窗口**来维护 cnt2​：滑动窗口每向右滑动一次，就多统计一次进入窗口的字符，少统计一次离开窗口的字符。然后，判断 cnt1​ 是否与 cnt2​ 相等，若相等则意味着 s1​ 的排列之一是 s2s 的子串。

也可以用
from collections import Counter
cnt1 = Counter(s1)
cnt2 = Counter(s2[ l : r ])