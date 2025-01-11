
**样例 1：**
输入：
```
nums = [1, 3, -1, 2, -1, 2]
```
输出：
```
7
```
解释：
最大的子数组为 [1, 3] 和 [2, -1, 2] 或者 [1, 3, -1, 2] 和 [2].  

**样例 2：**
输入：
```
nums = [5,4]
```
输出：
```
9
```
解释：
最大的子数组为 [5] 和 [4].


```python
import sys

class Solution:
    """
    @param: nums: A list of integers
    @return: An integer denotes the sum of max two non-overlapping subarrays
    """
    def max_two_sub_arrays(self, nums):
        n = len(nums)
        
        # 计算以i位置为结尾的前后缀最大连续和
        left_max = nums[:]
        right_max = nums[:]
        
        for i in range(1, n):
            left_max[i] = max(nums[i], left_max[i - 1] + nums[i])

        for i in range(n - 2, -1, -1):
            right_max[i] = max(nums[i], right_max[i + 1] + nums[i])
        
        # 计算前后缀部分最大连续和
        prefix_max = left_max[:]
        postfix_max = right_max[:]
    
        for i in range(1, n):
            prefix_max[i] = max(prefix_max[i], prefix_max[i - 1])
            
        for i in range(n - 2, -1, -1):
            postfix_max[i] = max(postfix_max[i], postfix_max[i + 1])
        
        result = -sys.maxsize
        for i in range(n - 1):
            result = max(result, prefix_max[i] + postfix_max[i + 1])
        
        return result
```
pass

