
样例

`n=4`，返回 `true`;

`n=5`，返回 `false`.


```python
    def check_power_of2(self, n):
        ans = 1
        for i in range(31):
            if ans == n:
                return True
            ans = ans << 1
 
        return False
```
pass