
**样例 1：**
```
输入：
a = "0", b = "0"
输出：
"0"
```
**样例 2：**
```
输入：
a = "11", b = "1"
输出：
"100"
```


```python
    def add_binary(self, a, b):
        indexa = len(a) - 1
        indexb = len(b) - 1
        carry = 0
        sum = ""
        while indexa >= 0 or indexb >= 0:
            x = int(a[indexa]) if indexa >= 0 else 0
            y = int(b[indexb]) if indexb >= 0 else 0
            if (x + y + carry) % 2 == 0:
                sum = '0' + sum
            else:
                sum = '1' + sum
            carry = (x + y + carry) // 2
            indexa, indexb = indexa - 1, indexb - 1
        if carry == 1:
            sum = '1' + sum
        return sum
```
pass