**lintcode 12 Min Stack带最小值操作的栈**

Example:

输入：
push(1)
min()
push(2)
min()
push(3)
min()
输出：
1
1
1

```python
class MinStack:
 
    def __init__(self):
        self.stack = []
        self.min_stack = []
 
    """
    @param: number: An integer
    @return: nothing
    """
    def push(self, number):
        self.stack.append(number)
        if len(self.min_stack) == 0 or number < self.min_stack[len(self.min_stack) - 1]:
            self.min_stack.append(number)
        else:
            self.min_stack.append(self.min_stack[len(self.min_stack) - 1])
 
    """
    @return: An integer
    """
    def pop(self):
        self.min_stack.pop()
        return self.stack.pop()
 
    """
    @return: An integer
    """
    def min(self):
        return self.min_stack[len(self.min_stack) - 1]
```
pass