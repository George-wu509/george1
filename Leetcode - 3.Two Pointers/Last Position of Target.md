

**样例 1：**
```
输入：nums = [1,2,2,4,5,5], target = 2
输出：2
```
**样例 2：**
```
输入：nums = [1,2,2,4,5,5], target = 6
输出：-1
```

```python
    def last_position(self, nums, target):
        if not nums or target is None:
            return -1

        start = 0
        end = len(nums) - 1

        while start + 1 < end:
            mid = start + (end - start) // 2

            if nums[mid] < target:
                start = mid
            elif nums[mid] > target:
                end = mid
            else:
                start = mid
    
        if nums[end] == target:
            return end
        elif nums[start] == target:
            return start
        else:
            return -1
```
pass