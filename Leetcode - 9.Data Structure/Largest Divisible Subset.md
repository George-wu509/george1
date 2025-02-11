Lintcode 603
给一个由 `无重复的正整数` 组成的集合，找出一个元素最多的子集，满足集合中任意两个元素 `(Si, Sj)` 都有 `Si % Sj = 0` 或 `Sj % Si = 0`

例1:
```python
"""
输入: nums =  [1,2,3], 
输出: [1,2] or [1,3]
```
例2:
```python
"""
输入: nums = [1,2,4,8], 
输出: [1,2,4,8]
```


```python
    def largest_divisible_subset(self, nums):
        if not nums:
            return []
            
        nums = sorted(nums)
        n = len(nums)
        dp, prev = {}, {}
        for num in nums:
            dp[num] = 1
            prev[num] = -1
        
        last_num = nums[0]
        for num in nums:
            for factor in self.get_factors(num):
                if factor not in dp:
                    continue
                if dp[num] < dp[factor] + 1:
                    dp[num] = dp[factor] + 1
                    prev[num] = factor
            if dp[num] > dp[last_num]:
                last_num = num
        
        return self.get_path(prev, last_num)
    
    def get_path(self, prev, last_num):
        path = []
        while last_num != -1:
            path.append(last_num)
            last_num = prev[last_num]
        return path[::-1]
        
    def get_factors(self, num):
        if num == 1:
            return []
        factor = 1
        factors = []
        while factor * factor <= num:
            if num % factor == 0:
                factors.append(factor)
                if factor * factor != num and factor != 1:
                    factors.append(num // factor)
            factor += 1
        return factors
```
pass