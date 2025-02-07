
Example:
**样例 1：**
输入：
```python
#nums = []
```
输出：
```python
#[]
```

**样例 2：**
输入：
```python
#nums = [1,1,2]
```
输出：
```python
[#[1,2]
```




```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        fast = slow = 1
        while fast < n:
            if nums[fast] != nums[fast - 1]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        
        return slow
```
pass