

**样例1**
```
输入: [5,2,4,3,1,6]
输出: 3
解释:
你可以把数组拆分为 : [5,2,1], [4,3], [6]。 这里总共有可以得到 3 个子序列。
或者你也可以拆分为 [5, 4, 3],[2,1], [6]。同样也是 3 个子序列。
但是 [5, 4, 3, 2, 1], [6] 是不合法的，因为 [5, 4, 3, 2, 1] 不是原数组的一个子序列。
```
**样例2**
```
输入: [1, 1, 1]
输出: 3
解释: 由于需要严格递减，所以必须拆分为 3 个子序列: [1],[1],[1]
```



```python
from typing import (
    List,
)

class Solution:
    # 二分查找数组中第一个大于 v 的元素的下标
    def upper_bound(self, nums: List[int], l: int, r: int, v: int) -> int:
        if v >= nums[r]:
            return r + 1
        while l < r:
            m = (l + r) >> 1
            if nums[m] <= v:
                l = m + 1
            else:
                r = m
        return l
    """
    @param array_in: The original array.
    @return: Count the minimum number of subarrays.
    """
    def least_subsequences(self, array_in: List[int]) -> int:
        # 表示每个子序列的最后一个数字。
        last = []
        # 遍历每个数字。
        for n in array_in:
            # 查找比当前数字大的拆分后的序列的最后一个数字。
            idx = 0
            if len(last) > 0:
                idx = self.upper_bound(last, 0, len(last) - 1, n)
            # 找不到，则新增一个拆分。
            if idx == len(last):
                last.append(n)
            # 否则更新找到的拆分的最后一个数字为当前数字。
            else:
                last[idx] = n
        # 返回拆分数。
        return len(last)
```
pass