
**样例 1:**
```
输入: 数组 = [1,2,7,8,5], 查询 = [(0,4),(1,2),(2,4)]
输出: [23,9,20]
```
**样例 2:**

```
输入: 数组 = [4,3,1,2],  查询 = [(1,2),(0,2)]
输出: [4,8]
```



```python
from typing import (

    List,

)

from lintcode import (

    Interval,

)

  

class Solution:

    def intervalSum(self, A, queries):

        # write your code here

        self.n = len(A)

        self.A = [0 for _ in range(self.n)]

        self.bit = [0 for _ in range(self.n + 1)]

        for i in range(self.n):

            self.update(i, A[i])

        result = []

        for query in queries:

            result.append(self.getPresum(query.end) - self.getPresum(query.start - 1))

        return result

    def lowbit(self, x):

        return x & (-x)

    def update(self, index, val):

        delta = val - self.A[index]

        self.A[index] = val

        i = index + 1

        while i <= self.n:

            self.bit[i] += delta

            i += self.lowbit(i)

    def getPresum(self, index):

        presum = 0

        i = index + 1

        while i > 0:

            presum += self.bit[i]

            i -= self.lowbit(i)

        return presum
```
pass