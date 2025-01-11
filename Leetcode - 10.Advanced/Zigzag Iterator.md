
**样例1**
```
输入: v1 = [1, 2] 和 v2 = [3, 4, 5, 6]
输出: [1, 3, 2, 4, 5, 6]
解释:
一开始轮换遍历两个数组，当v1数组遍历完后，就只遍历v2数组了，所以返回结果:[1, 3, 2, 4, 5, 6]
```
**样例2**
```
输入: v1 = [1, 1, 1, 1] 和 v2 = [3, 4, 5, 6]
输出: [1, 3, 1, 4, 1, 5, 1, 6]
```



```python
class ZigzagIterator:

    # @param {int[]} v1 v2 two 1d vectors
    def __init__(self, v1, v2):
        # initialize your data structure here
        self.queue = [v for v in (v1, v2) if v]


    def _next(self):
        # Write your code here
        v = self.queue.pop(0)
        value = v.pop(0)
        if v:
            self.queue.append(v)
        return value


    def hasNext(self):
        # Write your code here
        return len(self.queue) > 0
```
pass