
**样例 1：**
输入：
```
n = 9
```
输出：
```
10
```
解释：
[1,2,3,4,5,6,8,9,10,....]，第9个丑数为10。

**样例 2：**
输入：
```
n = 1
```
输出：
```
1
```



```python
    def nth_ugly_number(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[1] = 1
        p2 = p3 = p5 = 1

        for i in range(2, n + 1):
            num2, num3, num5 = dp[p2] * 2, dp[p3] * 3, dp[p5] * 5
            dp[i] = min(num2, num3, num5)
            if dp[i] == num2:
                p2 += 1
            if dp[i] == num3:
                p3 += 1
            if dp[i] == num5:
                p5 += 1
        
        return dp[n]
```
pass