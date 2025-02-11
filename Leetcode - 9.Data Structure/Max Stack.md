
lintcode859
设计一个支持push，pop，top，peekMax和popMax操作的最大栈。

1. push(x) -- 将元素x添加到栈中。
2. pop() -- 删除栈中最顶端的元素并将其返回。
3. top() -- 返回栈中最顶端的元素。
4. peekMax() -- 返回栈中最大的元素。
5. popMax() -- 返回栈中最大的元素，并将其删除。如果有多于一个最大的元素，只删除最靠近顶端的一个元素。


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