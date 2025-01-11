
**样例1**
```
输入: nums = [-2, 0, 3, -5, 2, -1]
sumRange(0, 2)
sumRange(2, 5)
sumRange(0, 5)
输出:
1
-1
-3
解释: 
sumRange(0, 2) -> (-2) + 0 + 3 = 1
sumRange(2, 5) -> 3 + (-5) + 2 + (-1) = -1
sumRange(0, 5) -> (-2) + 0 + 3 + (-5) + 2 + (-1) = -3
```
**样例2**
```
输入: 
nums = [-4, -5]
sumRange(0, 0)
sumRange(1, 1)
sumRange(0, 1)
sumRange(1, 1)
sumRange(0, 0)
输出: 
-4
-5
-9
-5
-4
解释: 
sumRange(0, 0) -> -4
sumRange(1, 1) -> -5
sumRange(0, 1) -> (-4) + (-5) = -9
sumRange(1, 1) -> -5
sumRange(0, 0) -> -4
```


```python
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.array = [0] * len(nums)
        self.array[0] = nums[0]
        for index in range(1, len(nums)):
            self.array[index] = self.array[index-1] + nums[index]
        

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        if i - 1 < 0:
            return self.array[j]
        else:
            return self.array[j] - self.array[i-1]
```
pass