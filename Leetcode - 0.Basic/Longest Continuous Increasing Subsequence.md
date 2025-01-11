

**样例 1：**
```
输入：[5, 4, 2, 1, 3]
输出：4
解释：
给定 [5, 4, 2, 1, 3]，其最长上升连续子序列（LICS）为 [5, 4, 2, 1]，返回 4。
```
**样例 2：**
```
输入：[1, 5, 2, 3, 4]
输出：3
解释：
给定 [1, 5, 2, 3, 4]，其最长上升连续子序列
```

```python
    def longest_increasing_continuous_subsequence(self, a):
        if not a:
            return 0
        longest, incr, desc = 1, 1, 1
        for i in range(1, len(a)):
            if a[i] > a[i - 1]:
                incr += 1
                desc = 1
            elif a[i] < a[i - 1]:
                incr = 1
                desc += 1
            else:
                incr = 1
                desc = 1
            longest = max(longest, max(incr, desc))
            
        return longest
```
pass