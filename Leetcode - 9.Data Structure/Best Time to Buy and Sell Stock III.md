Lintcode 151
假设你有一个数组，它的第i个元素是一支给定的股票在第i天的价格。设计一个算法来找到最大的利润。你最多可以完成两笔交易。

_**样例 1**_
```python
"""
输入 : [4,4,6,1,1,4,2,5]
输出 : 6
```


```python
    def max_Profit(self, prices):
        n = len(prices)
        K = 2
        # corner case
        if n == 0:
            return 0
        # main part
        dp = [[0] * n for _ in range(K + 1)]
        for i in range(1, K + 1):
            max_diff = float('-inf')
            for j in range(1, n):
                max_diff = max(max_diff, dp[i - 1][j - 1] - prices[j - 1])
                dp[i][j] = max(dp[i][j - 1], prices[j] + max_diff)
        return dp[K][n - 1]
```
pass