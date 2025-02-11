Lintcode 1280
给定一个非负整数的数据流输入 a1，a2，…，an，…，将到目前为止看到的数字总结为不相交的区间列表。

例如，假设数据流中的整数为 1，3，7，2，6，…，每次的总结为：

```python
"""
[1, 1]
[1, 1], [3, 3]
[1, 1], [3, 3], [7, 7]
[1, 3], [7, 7]
[1, 3], [6, 7]
```


**Example 1:**
```python
"""
输入：
addNum(1)
getIntervals()
addNum(3)
getIntervals()
addNum(7)
getIntervals()
addNum(2)
getIntervals()
addNum(6)
getIntervals()
输出：
[[(1,1)],[(1,1),(3,3)],[(1,1),(3,3),(7,7)],[(1,3),(7,7)],[(1,3),(6,7)]]
解释：
addNum(1)
getIntervals([[1, 1]])
addNum(3)
getIntervals([[1, 1], [3, 3]])
addNum(7)
getIntervals([[1, 1], [3, 3], [7, 7]])
addNum(2)-merge(1,2,3)
getIntervals([[1, 3], [7, 7]])
addNum(6)->merge(6,7)
getIntervals([[1, 3], [6, 7]])
```
**Example 2:**
```python
"""
输入：
addNum(1)
getIntervals([[1, 1]])
addNum(3)
getIntervals([[1, 1], [3, 3]])
输出：
[null,null,[4,4],null,[3,4]]
解释：
addNum(4)
getIntervals([[4, 4]])
addNum(3)->merge(3,4)
getIntervals([[3, 4]])
```


```python
class Solution:
    
    def __init__(self):
        self.father = dict()
        self.val2interval = dict()

    def addNum(self, val):
        if val in self.father:
            return
        
        self.father[val] = val
        self.val2interval[val] = [val, val]
        if val - 1 in self.father:
            self.merge(val - 1, val)
            
        if val + 1 in self.father:
            self.merge(val + 1, val)
      
    def getIntervals(self):
        return [
            Interval(self.val2interval[val][0], self.val2interval[val][1])
            for val in sorted(self.father.keys())
            if self.father[val] == val
        ]
        
    def merge(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        self.father[root_a] = root_b
        self.val2interval[root_b] = [
            min(self.val2interval[root_a][0], self.val2interval[root_b][0]),
            max(self.val2interval[root_a][1], self.val2interval[root_b][1]),
        ]
    
    def find(self, a):
        if a == self.father[a]:
            return a
        self.father[a] = self.find(self.father[a])
        return self.father[a]
```
pass