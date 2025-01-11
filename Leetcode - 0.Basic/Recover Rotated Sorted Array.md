
**样例 1：**
输入：
```
数组 = [4,5,1,2,3]
```
输出：
```
[1,2,3,4,5]
```
解释：
恢复旋转排序数组。

**样例 2：**
输入：
```
数组 = [6,8,9,1,2]
```
输出：
```
[1,2,6,8,9]
```
解释：
恢复旋转排序数组。


```python
class Solution:
    """
    @param nums: An integer array
    @return: nothing
    """
    def recover_rotated_sorted_array(self, nums):
        split_position = self.find_split(nums)
        if split_position == len(nums)-1:
            return 
        
        self.swap(nums, 0, split_position)
        self.swap(nums, split_position, len(nums))
        
        nums.reverse()
        return 
        
    def find_split(self, nums):
        # DO NOT use binary search!
        # Binary Search does not work on this prob
        if nums is None or len(nums) < 2:
            return 0
        
        for i in range(1,len(nums)):
            if nums[i] < nums[i-1]:
                return i 
        # return i = len()-1 if it's already a sorted array 
        return i 
            
    def swap(self, nums, start, end):
        if start == end:
            return nums 
        
        left, right = start, end -1  
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1 
            right -= 1
```
pass