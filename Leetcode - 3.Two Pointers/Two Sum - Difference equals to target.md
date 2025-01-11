
例1:
```
输入: nums = [2, 7, 15, 24], target = 5 
输出: [2, 7] 
解释:
(7 - 2 = 5)
```
例2:
```
输入: nums = [1, 1], target = 0
输出: [1, 1] 
解释:
(1 - 1 = 0)
```


```python
    def two_Sum7(self, nums, target):
        n = len(nums)
        if target < 0:
            target = -target
        j = 0
        for i in range(n):
            if i == j:
                j += 1
            while j < n and nums[j] - nums[i] < target:
                j += 1
            if j < n and nums[j] - nums[i] == target:
                return [nums[i],nums[j]]
```
pass