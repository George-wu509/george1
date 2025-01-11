

**样例 1**
输入：
```plaintext
3
["get()", "check(0)", "get()", "get()", "release(2)", "check(2)", "get()", "check(2)"]
```
输出：
```plaintext
[0, false, 1, 2, null, true, 2, false]
```

**样例 2**
输入：
```plaintext
0
["get()", "check(0)", "release(0)"]
```
输出：
```plaintext
[-1, false, null]
```



```python
class PhoneDirectory:

    def __init__(self, maxNumbers: int):
        self.available = [True] * maxNumbers

    """
    @return: the available number of the phone directory 
    """
    def get(self) -> int:
        for i in range(len(self.available)):
            if self.available[i]:
                self.available[i] = False
                return i
        return -1


    """
    @param number: the number to be checked
    @return: check whether the number of the phone directory is available
    """
    def check(self, number: int) -> bool:
        if number < 0 or number >= len(self.available):
            return False
        return self.available[number]


    """
    @param number: the number of the phone directory to be released
    @return: nothing
    """
    def release(self, number: int):
        if number < 0 or number >= len(self.available):
            return
        self.available[number] = True
```
pass