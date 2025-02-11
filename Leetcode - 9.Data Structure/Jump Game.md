Lintcode 116
给出一个非负整数数组，你最初定位在数组的第一个位置。

数组中的每个元素代表你在那个位置可以跳跃的最大长度。

判断你是否能到达数组的最后一个位置。

**样例 1：**
输入：
```python
"""
A = [2,3,1,1,4]
```
输出：
```python
"""
true
```
解释：
0 -> 1 -> 4（这里的数字为下标）是一种合理的方案。

**样例 2：**
输入：
```python
"""
A = [3,2,1,0,4]
```
输出：
```python
"""
false
```

```python
    def can_jump(self, a):
        if not a:
            return False

        n = len(a)
        # state: dp[i] 代表能否跳到坐标 i
        dp = [False] * n
        
        # initialization: 一开始站在0这个位置
        dp[0] = True
        
        # function
        for i in range(1, n):
            for j in range(i):
                # 高效的写法:
                if dp[j] and j + a[j] >= i:
                    dp[i] = True
                    break
                # 偷懒的写法
                # dp[i] = dp[i] or dp[j] and (j + a[j] >= i)
        
        # answer
        return dp[n - 1]
```
pass