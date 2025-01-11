
**样例 1：**
输入：
```
nums = []
k = 9
```
输出：
```
0
```
解释：
空数组，输出0

**样例 2：**
输入：
```
nums = [3,2,2,1]
k = 2
```
输出：
```
1
```
解释：
真实的数组为[1,2,2,3].所以返回 1

```python
class Solution:
    """
    @param nums: The integer array you should partition
    @param k: An integer
    @return: The index after partition
    """
    def partition_array(self, nums, k):
        left, right = 0, len(nums) - 1
        
        while left <= right:
            while left <= right and nums[left] < k:
                left += 1
            while left <= right and nums[right] >= k:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        
        return left
```
pass