

例1:
```
输入: nums = [0, 1, 0, 3, 12],
输出: [1, 3, 12, 0, 0].
```
例2:
```
输入: nums = [0, 0, 0, 3, 1],
输出: [3, 1, 0, 0, 0].
```


```python
    def move_zeroes(self, nums):
        left, right = 0, 0
        while right < len(nums):
            if nums[right] != 0:
                if left != right:
                    nums[left] = nums[right]
                left += 1
            right += 1
            
        while left < len(nums):
            if nums[left] != 0:
                nums[left] = 0
            left += 1
```
pass