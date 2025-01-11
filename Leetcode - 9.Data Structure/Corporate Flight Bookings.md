
样例 1：
```
输入：
bookings = [[1,2,10],[2,4,20],[2,5,25]]
n = 5
输出：
[10,55,45,45,25]
解释：
航班编号:       1    2    3    4    5
记录 1 :       10   10
记录 2 :            20   20   20
记录 3 :            25   25   25   25
总座位 :       10   55   45   45   25
```
样例 2：
```
输入：
bookings = [[1,2,10],[2,2,20]]
n = 2
输出：
[10,30]
解释：
航班编号:       1    2   
记录 1 :       10   10
记录 2 :            20 
总座位 :       10   30
```




```python
import itertools
class Solution:

    def corp_flight_bookings(self, bookings, n):
        res = [0] * n
        for start, end, seats in bookings:
            res[start - 1] += seats
            if end < n:
                res[end] -= seats
        return list(itertools.accumulate(res))
```
pass