
**样例 1：**
输入：
```
A = "ABCD"
B = "EDCA"
```
输出：
```
1
```
解释：
LCS是 'A' 或 'D' 或 'C'  

**样例 2：**
输入：
```
A = "ABCD"
B = "EACB"
```
输出：
```
2
```
解释：
LCS 是 "AC" 或 "AB"


```python
    def longest_common_subsequence(self, a: str, b: str) -> int:
        if not a or not b:
            return 0
            
        n, m = len(a), len(b)
        # state & initialization
        dp = [[0] * (m + 1) for i in range(n + 1)]
        
        # function
        for i in range(1, n + 1):
            for j in range(1, m +  1):
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1)

        # answer
        return dp[n][m]
```
pass