89
给定 `n` 个不同的正整数，整数 `k`(k≤n)(k≤n)以及一个目标数字 `target`。在这 `n` 个数里面找出 `k` 个数，使得这 `k` 个数的和等于目标数字，求问有多少种方案？

Example:
**样例 1：**
输入：
```python
A = [1,2,3,4]
k = 2
target = 5
```
输出：
```python
2
```
解释：
1 + 4 = 2 + 3 = 5  

**样例 2：**
输入：
```python
A = [1,2,3,4,5]
k = 3
target = 6
```
输出：
```python
1
```
解释：
只有这一种方案。 1 + 2 + 3 = 6

```python
class Solution:
    def kSum(self, A, k, target):
        n = len(A)
        dp = [
            [[0] * (target + 1) for _ in range(k + 1)],
            [[0] * (target + 1) for _ in range(k + 1)],
        ]
        
        # dp[i][j][s]
        # 前 i 个数里挑出 j 个数，和为 s
		# Example: k=2, target = 3
        # dp = [
		#		[[0,0,0,0],[0,0,0,0],[0,0,0,0]]
		#		[[0,0,0,0],[0,0,0,0],[0,0,0,0]]]

        dp[0][0][0] = 1
        for i in range(1, n + 1):
            dp[i % 2][0][0] = 1
            for j in range(1, min(k + 1, i + 1)):
                for s in range(1, target + 1):
                    dp[i % 2][j][s] = dp[(i - 1) % 2][j][s]
                    if s >= A[i - 1]:
                        dp[i % 2][j][s] += dp[(i - 1) % 2][j - 1][s - A[i - 1]]
                        
        return dp[n % 2][k][target]
```
pass