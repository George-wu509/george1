

**例 1:**
```
输入: quality = [10,20,5], wage = [70,50,30], K = 2
输出: 105.00
解释: 花费70雇佣0号，花费35雇佣2号
```
**例 2:**
```
输入: quality = [3,1,10,10,1], wage = [4,8,2,2,7], K = 3
输出: 30.67
解释: 花费4雇佣0号，花费13.333雇佣2号和3号 
```



```python
import heapq

class Solution:

    def mincost_to_hire_workers(self, quality, wage, k):
        from fractions import Fraction
        workers = sorted((Fraction(w, q), q, w)
        for q, w in zip(quality, wage))

        ans = float('inf')
        pool = []
        sumq = 0
        for ratio, q, w in workers:
            heapq.heappush(pool, -q)
            sumq += q

            if len(pool) > k:
                sumq += heapq.heappop(pool)

            if len(pool) == k:
                ans = min(ans, ratio * sumq)
        return float(ans)
```
pass