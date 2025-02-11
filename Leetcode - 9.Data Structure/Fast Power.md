Lintcode 140
计算an%ban%b，其中`a`，`b`和`n`都是32位的非负整数。


**样例 1：**
输入：
```python
"""
a = 3
b = 7
n = 5
```
输出：
```python
"""
5
```
解释：
3 ^ 5 % 7 = 5

**样例 2：**
输入：
```python
"""
a = 3
b = 1
n = 0
```
输出：
```python
"""
0
```
解释：
3 ^ 0% 1 = 0


```python
    def fastPower(self, a, b, n):
        ans = 1
        while n > 0:
            if n % 2 == 1:
                ans = (ans * a) % b
            a = a * a % b
            n = n // 2
        return ans % b
```
pass