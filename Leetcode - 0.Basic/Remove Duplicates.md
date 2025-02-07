Lintcode 100
给定一个排序数组，在原数组中“删除”重复出现的数字，使得每个元素只出现一次，并且返回新数组。不要使用额外的数组空间，必须在不使用额外空间的条件下原地完成。我们会通过返回的数组长度 k，截取数组前 k 个元素来判断正确性。

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

解釋:
step1. 因為是排序數組, 所以用一個指針從id=0到最後, 檢查相鄰數字是有相同就好

Time complexity should be O(n)