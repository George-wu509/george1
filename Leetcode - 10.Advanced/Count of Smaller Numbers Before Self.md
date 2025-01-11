

**样例 1:**
```
输入:
[1,2,7,8,5]
输出:
[0,1,2,3,2]
```
**样例 2:**
```
输入:
[7,8,2,1,3]
输出:
[0,1,0,0,2]
```



```python
from typing import (
    List,
)

class BITree:
    def __init__(self, num_range):
        self.bit = [0] * (num_range + 1)
    
    def get_prefix_sum(self, index):
        result = 0
        index = index + 1
        while index > 0:
            result += self.bit[index]
            index -= self._lowbit(index)
        
        return result
    
    def increase_count(self, index, delta):
        index = index + 1
        while index < len(self.bit):
            self.bit[index] += delta
            index += self._lowbit(index)
    
    def _lowbit(self, x):
        return x & (-x)

class Solution:

    def countOfSmallerNumberII(self, A):
        bitree = BITree(10000)
        results = []
        
        for num in A:
            smaller_count = bitree.get_prefix_sum(num - 1)
            results.append(smaller_count)
            bitree.increase_count(num, 1)
        
        return results
```
pass