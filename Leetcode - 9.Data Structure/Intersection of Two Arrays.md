Lintcode 547
给出两个数组，写出一个方法求出它们的交集


**例1:**
```python
"""
输入: nums1 = [1, 2, 2, 1], nums2 = [2, 2], 
输出: [2].
```
**例2:**
```python
"""
输入: nums1 = [1, 2], nums2 = [2], 
输出: [2].
```


```python
    def intersection(self, nums1, nums2):
        s1, s2 = set(nums1), set(nums2)
        return [x for x in s1 if x in s2]
```
pass