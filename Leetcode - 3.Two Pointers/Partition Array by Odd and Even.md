lintcode 373
分割一个整数数组，使得奇数在前偶数在后。

**样例1:**
```python
"""```python
输入：
[1, 2, 3, 4]
输出：
[1, 3, 2, 4]
```
**样例2:**

```python
"""
输入：
[1, 4, 2, 3, 5, 6]
输出： 
[1, 3, 5, 4, 2, 6]
解释：
答案不唯一，另一种可行的解是 [1, 5, 3, 2, 4
```


```python
    def partition_array(self, nums):
        start, end = 0, len(nums) - 1
        while start <= end:
            while start <= end and nums[start] % 2 == 1:
                start += 1
            while start <= end and nums[end] % 2 == 0:
                end -= 1
            if start <= end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
```
pass