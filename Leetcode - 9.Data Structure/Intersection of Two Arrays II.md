Lintcode 548
给定两个数组，计算两个数组的交集

**样例1**
```python
"""
输入: 
nums1 = [1, 2, 2, 1], nums2 = [2, 2]
输出: 
[2, 2]
```
**样例2**
```python
"""
输入: 
nums1 = [1, 1, 2], nums2 = [1]
输出: 
[1]
```


```python
    def intersection(self, nums1, nums2):
        counts = collections.Counter(nums1)
        result = []

        for num in nums2:
            if counts[num] > 0:
                result.append(num)
                counts[num] -= 1

        return result
```
pass