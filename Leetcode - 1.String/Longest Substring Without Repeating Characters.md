

**样例 1:**
```
输入: "abcabcbb"
输出: 3
解释: 最长子串是 "abc".
```
**样例 2:**
```
输入: "bbbbb"
输出: 1
解释: 最长子串是 "b".
```

```python
    def length_of_longest_substrin(self, s):
        unique_chars = set([])
        j = 0
        n = len(s)
        longest = 0
        for i in range(n):
            while j < n and s[j] not in unique_chars:
                unique_chars.add(s[j])
                j += 1
            longest = max(longest, j - i)
            unique_chars.remove(s[i])
            
        return longest
```
pass