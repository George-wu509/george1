
lintcode859 最大栈 Max Stack

Example
输入:
push(5)
push(1)
push(5)
top()
popMax()
top()
peekMax()
pop()
top()
输出:
5, 5, 1, 5, 1, 5

```python
class MaxStack(list):
    def push(self, x):
        # 如果栈为空，直接将 x 作为最大值
        m = x if not self else max(x, self[-1][1])
        self.append((x, m))

    def pop(self):
        return list.pop(self)[0]

    def top(self):
        return self[-1][0]

    def peekMax(self):
        return self[-1][1]

    def popMax(self):
        m = self[-1][1]
        buffer = []
        # 将元素弹出直到找到最大值
        while self[-1][0] != m:
            buffer.append(self.pop())
        # 移除最大值
        self.pop()
        # 将其余元素重新压回栈
        for x in reversed(buffer):
            self.push(x)
        return m
```
pass