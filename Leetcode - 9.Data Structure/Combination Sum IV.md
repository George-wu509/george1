

**样例1**
```
输入: nums = [1, 2, 4] 和 target = 4
输出: 6
解释:
可能的所有组合有：
[1, 1, 1, 1]
[1, 1, 2]
[1, 2, 1]
[2, 1, 1]
[2, 2]
[4]
```
**样例2**
```
输入: nums = [1, 2] 和 target = 4
输出: 5
解释:
可能的所有组合有：
[1, 1, 1, 1]
[1, 1, 2]
[1, 2, 1]
[2, 1, 1]
[2, 2]
```




```python
    def back_pack_v_i(self, nums: List[int], target: int) -> int:
        dp = [1] + [0] * target
        for i in range(1, target + 1):
            for num in nums:
                if num <= i:
                    dp[i] += dp[i - num]
        
        return dp[target]
```
pass