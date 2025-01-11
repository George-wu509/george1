
**样例1**
```
输入: list = [[1,1],2,[1,1]]
输出: [1,1,2,1,1]
```
**样例2**
```
输入: list = [1,[4,[6]]]
输出: [1,4,6]
```



```python
class NestedIterator(object):

    def __init__(self, nestedList):
        self.next_elem = None
        self.stack = []
        for elem in reversed(nestedList):
            self.stack.append(elem)
            
    # @return {int} the next element in the iteration
    def next(self):
        if self.next_elem is None:
            self.hasNext()
        temp, self.next_elem = self.next_elem, None
        return temp
        
    # @return {boolean} true if the iteration has more element or false
    def hasNext(self):
        if self.next_elem:
            return True
            
        while self.stack:
            top = self.stack.pop()
            if top.isInteger():
                self.next_elem = top.getInteger()
                return True
            for elem in reversed(top.getList()):
                self.stack.append(elem)
        return False
```
pass