
**样例 1：**
输入：
```
word1 = "horse"
word2 = "ros"
```
输出：
```
3
```
解释：
horse -> rorse (替换 'h' 为 'r')  
rorse -> rose (删除 'r')  
rose -> ros (删除 'e')

**样例 2：**
输入：
```
word1 = "intention"
word2 = "execution"
```
输出：
```
5
```
解释：
intention -> inention (删除 't')  
inention -> enention (替换 'i' 为 'e')  
enention -> exention (替换 'n' 为 'x')  
exention -> exection (替换 'n' 为 'c')  
exection -> execution (插入 'u')


```python
class Solution: 
    def min_distance(self, word1, word2):
        if word1 is None or word2 is None:
            return -1
            
        n, m = len(word1), len(word2)
        # state
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # initialization
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        # function
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1
                if word2[j - 1] == word1[i - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])
                else:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1)
                    
        # answer
        return dp[n][m]
```
pass