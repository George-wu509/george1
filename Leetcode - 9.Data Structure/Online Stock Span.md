
**样例 1:**
```
输入：prices = [100,80,60,70,60,75,85]
输出：[1,1,1,2,1,4,6]
解释：
首先，初始化 S = StockSpanner()，然后：
S.next(100) 被调用并返回 1，
S.next(80) 被调用并返回 1，
S.next(60) 被调用并返回 1，
S.next(70) 被调用并返回 2，
S.next(60) 被调用并返回 1，
S.next(75) 被调用并返回 4，
S.next(85) 被调用并返回 6。

注意 (例如) S.next(75) 返回 4，因为截至今天的最后 4 个价格
(包括今天的价格 75) 小于或等于今天的价格。
```
**样例 2:**

```
输入：prices = [50,80,80,70,90,75,85]
输出：[1,2,3,1,5,1,2]
解释：
首先，初始化 S = StockSpanner()，然后：
S.next(50) 被调用并返回 1，
S.next(80) 被调用并返回 2，
S.next(80) 被调用并返回 3，
S.next(70) 被调用并返回 1，
S.next(90) 被调用并返回 5，
S.next(75) 被调用并返回 1，
S.next(85) 被调用并返回 2。
```


```python
class StockSpanner(object):
    def __init__(self):
        self.stack = []

    def next(self, price):
        weight = 1
        while self.stack and self.stack[-1][0] <= price:
            weight += self.stack.pop()[1]
        self.stack.append((price, weight))
        return weight
```
pass