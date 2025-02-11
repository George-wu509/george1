Lintcode 3659
在本题中，你需要设计一个电话目录管理系统类。

该电话目录管理系统的构造函数会接收一个整数类型的变量 `maxNumbers` ，该变量表示电话目录的最大编号。

除了构造函数以外，你还需要合理设计该类中的以下函数：

- `int get()`：获取电话目录中下一个可用编号，如果不存在可用编号则返回 -1
- `bool check(int number)`：检查指定编号是否可用
- `void release(int number)`：修改某个编号的状态为可用


**样例 1**
输入：
```python
"""
3
["get()", "check(0)", "get()", "get()", "release(2)", "check(2)", "get()", "check(2)"]
```
输出：
```python
"""
[0, false, 1, 2, null, true, 2, false]
```

**样例 2**
输入：
```python
"""
0
["get()", "check(0)", "release(0)"]
```
```python
"""
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