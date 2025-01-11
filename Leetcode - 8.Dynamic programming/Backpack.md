
**样例 1：**
输入：
```
数组 = [3,4,8,5]
backpack size = 10
```
输出：
```
9
```
解释：
装4和5.

**样例 2：**
输入：
```
数组 = [2,3,5,7]
backpack size = 12
```
输出：
```
12
```
解释：
装5和7.


```python
class Solution:
    def back_pack(self, m, a):
        n = len(a)
        f = [[False] * (m + 1) for _ in range(n + 1)]
        
        f[0][0] = True
        for i in range(1, n + 1):
            f[i][0] = True
            for j in range(1, m + 1):
                if j >= a[i - 1]:
                    f[i][j] = f[i - 1][j] or f[i - 1][j - a[i - 1]]
                else:
                    f[i][j] = f[i - 1][j]
                    
        for i in range(m, -1, -1):
            if f[n][i]:
                return i
        return 0
```