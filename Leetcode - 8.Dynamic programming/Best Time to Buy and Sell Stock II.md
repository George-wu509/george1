
**样例 1:**
```
输入: [2, 1, 2, 0, 1]
输出: 2
解释: 
    1. 在第 2 天以 1 的价格买入, 然后在第 3 天以 2 的价格卖出, 利润 1
    2. 在第 4 天以 0 的价格买入, 然后在第 5 天以 1 的价格卖出, 利润 1
    总利润 2.
```
**样例 2:**
```
输入: [4, 3, 2, 1]
输出: 0
解释: 不进行任何交易, 利润为0.
```


```python
class Solution:
    def max_profit(self, prices: List[int]) -> int:
        if len(prices) == 0:
            return 0
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0], dp[0][1] = 0, -prices[0]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[n - 1][0]
```
pass