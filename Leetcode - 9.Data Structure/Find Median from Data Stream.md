
**样例 1：**
输入：
```
add(1)
getMedian()
add(2)
getMedian()
add(3)
getMedian()
add(4)
getMedian()
add(5)
getMedian()
```
输出：
```
1
1
2
2
3
```
解释：
[1] 和 [1,2] 的中位数是 1，  
[1,2,3] 和 [1,2,3,4] 的中位数是 2，  
[1,2,3,4,5] 的中位数是 3。  

**样例 2：**
输入：
```
add(4)
getMedian()
add(5)
getMedian()
add(1)
getMedian()
add(3)
getMedian()
add(2)
getMedian()
add(6)
getMedian()
add(0)
getMedian()
```
输出：
```
4
4
4
3
3
3
3
```
解释：
[4], [4,5] 和 [4,5,1] 的中位数是 4，  
[4,5,1,3], [4,5,1,3,2], [4,5,1,3,2,6] 和 [4,5,1,3,2,6,0] 的中位数是 3。




```python
import heapq
class Solution:
    def __init__(self):
        self.max_heap = []
        self.min_heap = []
        self.is_first_add = True

    def add(self, val):
        if self.is_first_add:
            self.median = val
            self.is_first_add = False
            return
    
        if val < self.median:
            heapq.heappush(self.max_heap, -val)
        else:
            heapq.heappush(self.min_heap, val)

        if len(self.max_heap) > len(self.min_heap):
            heapq.heappush(self.min_heap, self.median)
            self.median = -heapq.heappop(self.max_heap)
        if len(self.max_heap) < len(self.min_heap) - 1:
            heapq.heappush(self.max_heap, -self.median)
            self.median = heapq.heappop(self.min_heap)

    def getMedian(self):
        return self.median
```
pass