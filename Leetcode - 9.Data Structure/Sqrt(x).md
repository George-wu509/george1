Lintcode 141
实现 `int sqrt(int x)` 函数，计算并返回 _x_ 的平方根。

```python
"""
样例 1:
输入:  0
	输出: 0


样例 2:
	输入: 3
	输出: 1
	
	样例解释：
	返回对x开根号后向下取整的结果。

样例 3:
	输入: 4
	输出: 2
```


```python
    def sqrt(self, x: int) -> int:
        if x == 0:
            return 0
        
        C, x0 = float(x), float(x)
        while True:
            xi = 0.5 * (x0 + C / x0)
            if abs(x0 - xi) < 1e-7:
                break
            x0 = xi
        
        return int(x0)
```
pass