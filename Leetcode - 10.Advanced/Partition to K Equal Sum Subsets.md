
**样例1**
```
输入: nums = [4, 3, 2, 3, 5, 2, 1] 和 k = 4
输出: True
解释:
一个可能的划分 (5), (1, 4), (2, 3), (2, 3)拥有相等的权重
```
**样例2**
```
输入: nums = [4, 3, 2, 3, 5, 2, 1] 和 k = 3
输出: True
解释:
一个可能的划分 (1, 2, 3), (1, 5), (3, 3)拥有相等的权重
```



```python
class Solution:
    def partitionto_equal_sum_subsets(self, nums: List[int], k: int) -> bool:
        if k == 1:
            return True
        l = len(nums)
        nums.sort()
        sums = sum(nums)
        if sums % k != 0:
            return False
        target = sums // k
        if nums[-1] > target:
            return False
        
        size = 1 << l
        dp = [0] * size
        dp[0] = 1
        current_sum = [0] * size
        for i in range(size):
            if not dp[i]:
                continue
            for j in range(l):
                if i & (1 << j) != 0:
                    continue
                nex = i | ( 1 << j)
                if dp[nex]:
                    continue
                if current_sum[i] % target + nums[j] <= target:
                    current_sum[nex] = current_sum[i] + nums[j]
                    dp[nex] = 1
                else:
                    break
        return dp[-1] == 1
```
pass