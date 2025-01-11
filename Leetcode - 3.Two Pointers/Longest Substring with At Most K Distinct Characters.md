

**样例 1:**
```
输入: S = "eceba" 并且 k = 3
输出: 4
解释: T = "eceb"
```
**样例 2:**
```
输入: S = "WORLD" 并且 k = 4
输出: 4
解释: T = "WORL" 或 "ORLD"
```


```python
    def length_of_longest_substring_k_distinct(self, s, k):
        if not s:
            return 0
            
        counter = {}
        left = 0
        longest = 0
        for right in range(len(s)):
            counter[s[right]] = counter.get(s[right], 0) + 1
            while left <= right and len(counter) > k:
                counter[s[left]] -= 1
                if counter[s[left]] == 0:
                    del counter[s[left]]
                left += 1
            
            longest = max(longest, right - left + 1)
        return longest
```
pass