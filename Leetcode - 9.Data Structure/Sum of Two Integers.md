Lintcode 1276
不使用运算符 + 和 - ，计算两整数 a 、b 之和。


**样例 1:**
```python
"""
输入: a = 1, b = 2
输出: 3
```
**样例 2:**
```python
"""
输入: a = -2, b = 3
输出: 1
```


```python
class Solution:
    MASK1 = 4294967296  # 2^32
    MASK2 = 2147483648  # 2^31
    MASK3 = 2147483647  # 2^31-1

    def get_sum(self, a: int, b: int) -> int:
        a %= self.MASK1
        b %= self.MASK1
        while b != 0:
            carry = ((a & b) << 1) % self.MASK1
            a = (a ^ b) % self.MASK1
            b = carry
        if a & self.MASK2:  # 负数
            return ~((a ^ self.MASK2) ^ self.MASK3)
        else:  # 正数
            return a
```
pass