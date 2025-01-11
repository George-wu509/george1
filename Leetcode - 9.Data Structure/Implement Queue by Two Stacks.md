

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