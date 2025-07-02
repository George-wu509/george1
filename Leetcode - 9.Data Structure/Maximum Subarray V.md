
Lintcode 621
给定一个整数数组，找到长度在 `k1` 与 `k2` 之间(包括 k1, k2)的子数组并且使它们的和最大，返回这个最大值，如果数组元素个数小于 k1 则返回 0


```python
from collections import deque
class Solution:

    def maxSubarray5(self, nums, k1, k2):
        n = len(nums)
        prefix = [0 for _ in xrange(n + 1)]
        curMax = -float('inf')
        dq = deque([])
        curMin = 0
        for i in xrange(1, n + 1):
            prefix[i] = prefix[i-1] + nums[i-1]
            while dq and i - dq[0] > k2:
                dq.popleft()
            if i >= k1:
                while dq and prefix[dq[-1]] > prefix[i - k1]:
                    dq.pop()
                dq.append(i - k1)
            if dq and k1 <= i - dq[0] <= k2:
               curMax = max(curMax, prefix[i] - prefix[dq[0]])
        return curMax if curMax != -float('inf') else 0

```




首先建立prefix sum  
然後用deque維持一個最大長度為k2 prefix遞增的數列 存的數為index

當i - k1 >= 0時, 表示長度已經大於k1了, 可以開始有解  
因此把i - k1這個index加到deque裡面  
但其實我們知道, 求maxsubarry只需要現在的prefix減去最小值  
因此可以用一個while loop, 把所有比要加上去的數值(prefix[i-k])小的數都pop掉  
最後如果deque裡有值, 就比較目前的最大值及prefix[i] - deque[0]得到答案

＝＝＝＝舉個例子會清楚很多＝＝＝＝  
如果今天題目是[-2,2,-3,4,-1,2,1,-5,3]  
prefix = [0,-2,0,-3,1,0,2,3,-2,1]  
k1, k2 = 2, 4  
i = 2: prefix[i] = 0, dq = [0], prefix[i-k] = 0, curMax = 0  
i = 3: prefix[i] = -3, dq = [1], prefix[i-k] = -2, curMax = -1 (-2比0小, 把0 pop掉)  
i = 4: prefix[i] = 1, dq = [1,2], prefix[i-k] = 0, curMax = 3  
i = 5: prefix[i] = 0, dq = [3], prefix[i-k] = -3, curMax = 3 (-3比-2跟0都小, 全部pop掉)  
i = 6: prefix[i] = 2, dq = [3,4], prefix[i-k] = 1, curMax = 5  
i = 7: prefix[i] = 3, dq = [3,5], prefix[i-k] = 0, curMax = 6 (0比1小, pop掉1)  
i = 8: prefix[i] = -2, dq = [5,6], prefix[i-k] = 2, curMax = -2 (i = 8, dq[0] = 3, 長度超過k2, popleft)  
i = 9: prefix[i] = 1, dq = [5,6,7], prefix[i-k] = 3, curMax = 1  
所以答案是6



Time: O(N), Space: O(k2)  
