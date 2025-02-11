Lintcode


**样例1**
```python
"""
输入: [3, 10, 1000, -99, 4, 100] 并且 k = 3
输出: [1000, 100, 10]
```
**样例2**
```python
"""
输入: [8, 7, 6, 5, 4, 3, 2, 1] 并且 k = 5
输出: [8, 7, 6, 5, 4]
```



```python
import heapq
class Solution:

    def topk(self, nums: List[int], k: int) -> List[int]:
        heapq.heapify(nums)
        topk = heapq.nlargest(k, nums)
        topk.sort()
        topk.reverse()
        return topk

```
pass