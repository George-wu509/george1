
**样例 1：**
输入：
```
nums = [5,4,1,2,3]
```
输出：
```
3
```
解释：
LIS 是 [1,2,3] 

**样例 2：**
输入：
```
nums = [4,2,4,5,3,7]
```
输出：
```
4
```
解释：
LIS 是 [2,4,5,7]


```python
class Solution:
    def longest_increasing_subsequence(self, nums):
        if nums is None or not nums:
            return 0
        
        # state: dp[i] 表示从左到右跳到i的最长sequence 的长度
        
        # initialization: dp[0..n-1] = 1
        dp = [1] * len(nums)
        
        # prev[i] 代表 dp[i] 的最优值是从哪个 dp[j] 算过来的
        prev = [-1] * len(nums)
        
        # function dp[i] = max{dp[j] + 1},  j < i and nums[j] < nums[i]
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i] and dp[i] < dp[j] + 1:
                    dp[i] = dp[j] + 1
                    prev[i] = j
        
        # answer: max(dp[0..n-1])
        longest, last = 0, -1
        for i in range(len(nums)):
            if dp[i] > longest:
                longest = dp[i]
                last = i
        
        path = []
        while last != -1:
            path.append(nums[last])
            last = prev[last]
        print(path[::-1])
        
        return longest
```
pass