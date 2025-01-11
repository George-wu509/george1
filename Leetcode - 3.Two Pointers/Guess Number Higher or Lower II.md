
**样例1**
```
输入： 10
输出： 16
解释：
给出 n = 10, 我选择的数为 2
第一轮: 你猜测为 7, 我告诉你待猜的值要更小一些. 你需要支付 $7
第二轮: 你猜测为 3, 我告诉你待猜的值要更小一些. 你需要支付 $3
第三轮: 你猜测为 1, 我告诉你待猜的值要更大一些. 你需要支付 $1
游戏结束. 2 是我选择的待猜数. 
你最终需要支付 $7 + $3 + $1 = $11

给出 n = 10, 我选择的数为 4
第一轮: 你猜测为 7, 我告诉你待猜的值要更小一些. 你需要支付 $7
第二轮: 你猜测为 3, 我告诉你待猜的值要更大一些. 你需要支付 $3
第三轮: 你猜测为 5, 我告诉你待猜的值要更小一些. 你需要支付 $5
游戏结束. 4 是我选择的待猜数. 
你最终需要支付 $7 + $3 + $5 = $15

给出 n = 10, 我选择的数为 8
第一轮: 你猜测为 7, 我告诉你待猜的值要更大一些. 你需要支付 $7
第二轮: 你猜测为 9, 我告诉你待猜的值要更小一些. 你需要支付 $9
游戏结束. 8 是我选择的待猜数. 
你最终需要支付 $7 + $9 = $16

所以对于 n = 10, 答案为 16.
```
**样例2**
```
输入： 5
输出： 6
```


```python
class Solution:
    def get_money_amount(self, n: int) -> int:
        dp = [[0 for _ in range(n + 1)] for __ in range(n + 1)]
        for len in range(2, n + 1):
            for start in range(1, n - len + 2):
                import sys
                temp = sys.maxsize
                for k in range(start + (len - 1) // 2, start + len - 1):
                    left, right = dp[start][k - 1], dp[k + 1][start + len - 1]
                    temp = min(k + max(left, right), temp)
                    if left > right:
                        break
                dp[start][start + len - 1] = temp

        return dp[1][n]
```
pass