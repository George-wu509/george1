
**样例 1:**
```
输入:
  nums = [1, 3, 5]
  sumRange(0, 2)
  update(1, 2)
  sumRange(0, 2)
输出: 
  9
  8
```
**样例 2:**
```
输入: 
  nums = [0, 9, 5, 7, 3]
  sumRange(4, 4)
  sumRange(2, 4)
  update(4, 5)
  update(1, 7)
  update(0, 8)
  sumRange(1, 2)
输出: 
  3
  15
  12
```



```python
class NumArray(object):
    
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.arr = nums
        self.n = len(nums)
        self.bit = [0] * (self.n + 1)
        for i in range(self.n):
            self.add(i, self.arr[i])

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: void
        """
        self.add(i, val - self.arr[i])
        self.arr[i] = val

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.sum(j) - self.sum(i - 1)
        
    def lowbit(self, x):
        return x & (-x)
    
    def add(self, idx, val):
        idx += 1
        while idx <= self.n:
            self.bit[idx] += val
            idx += self.lowbit(idx)
    
    def sum(self, idx):
        idx += 1
        res = 0
        while idx > 0:
            res += self.bit[idx]
            idx -= self.lowbit(idx)
        return res
```
pass