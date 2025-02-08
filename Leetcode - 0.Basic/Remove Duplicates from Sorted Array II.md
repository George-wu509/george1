Lintcode 101
给你一个排序数组 `nums`，在逻辑上删除其中的重复元素，返回新的数组的长度 `len`，使得原数组 `nums` 的前 `len` 个元素中，每个数字最多出现两次。如果一个数字出现超过2次，则这个数字最后保留两个。

**样例 1：**
输入：
```python
"""
#数组 = []
```
输出：
```python
"""
#0
```
解释：
空数组，长度为0.

**样例 2：**
输入：
```python
"""
#数组 = [1,1,1,2,2,3]
```
输出：
```python
"""
#5
```
解释：
长度为 5， 数组为：[1,1,2,2,3]


```python
    def removeDuplicates(self, nums):
        B = []
        before = None
        countb = 0
        for number in nums:
            if(before != number):
                B.append(number)
                before = number
                countb = 1
            elif countb < 2:
                B.append(number)
                countb += 1
        p = 0
        for number in B:
            nums[p] = number
            p += 1
        return p
```
pass