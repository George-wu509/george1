
**样例 1:**
```
输入: "PPALLP"
输出: True
```
**样例 2:**
```
输入: "PPALLL"
输出: False
```


```python
    def check_record(self, s: str) -> bool:
        absents = lates = 0
        for i, c in enumerate(s):
            if c == "A":
                absents += 1
                if absents >= 2:
                    return False
            if c == "L":
                lates += 1
                if lates >= 3:
                    return False
            else:
                lates = 0
        
        return True
```
pass