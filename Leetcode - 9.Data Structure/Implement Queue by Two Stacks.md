Lintcode 40
正如标题所述，你只能使用两个栈来实现队列的一些操作。队列应支持`push(element)`，`pop()` 和 `top()`，其中`pop`是弹出队列中的第一个(最前面的)元素。`pop`和`top`方法都应该返回第一个元素的值。

```python
class MyQueue:
    
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, element):
        self.stack1.append(element)

    def pop(self):
        if len(self.stack2) == 0:
            self.move()
        return self.stack2.pop()

    def top(self):
        if len(self.stack2) == 0:
            self.move()
        return self.stack2[-1]
    
    def move(self):
        while len(self.stack1) > 0:
            self.stack2.append(self.stack1.pop())
```