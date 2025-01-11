
**样例 1:**
```
输入: [2, 7, 11, 15], target = 24
输出: 1
解释: 11 + 15 是唯一的一对
```
**样例 2:**
```
输入: [1, 1, 1, 1], target = 1
输出: 6
```

```python
    def two_sum2(self, nums, target):
        n = len(nums)
        if n < 2:
            return 0
        
        nums.sort()
        
        res = 0
        l, r = 0, n - 1
        while l < r:
            if nums[l] + nums[r] <= target:
                l += 1
            else:
                res += r - l
                r -= 1
                
        return res
```
pass