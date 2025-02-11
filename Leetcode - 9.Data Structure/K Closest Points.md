Lintcode


例1:
```python
"""
输入：
points = [[4,6],[4,7],[4,4],[2,5],[1,1]]
origin = [0, 0]
k = 3 
输出：
[[1,1],[2,5],[4,4]]
```
例2:
```python
"""
输入：
points = [[0,0],[0,9]]
origin = [3, 1]
k = 1
输出：
[[0,0]]
```



```python
import heapq
class Solution:

    def k_closest(self, points: List[Point], origin: Point, k: int) -> List[Point]:
        self.heap = []
        for point in points:
            dist = self.getDistance(point, origin)
            heapq.heappush(self.heap, (-dist, -point.x, -point.y))
            
            if len(self.heap) > k:
                heapq.heappop(self.heap)

        ret = []
        while len(self.heap) > 0:
            _, x, y = heapq.heappop(self.heap)
            ret.append(Point(-x, -y))

        ret.reverse()
        return ret

    def getDistance(self, a, b):
        return (a.x - b.x) ** 2 + (a.y - b.y) ** 2
```
pass