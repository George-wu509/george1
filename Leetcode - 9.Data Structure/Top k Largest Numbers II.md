
**样例1**

```
输入: 
s = new Solution(3);
s.add(3)
s.add(10)
s.topk()
s.add(1000)
s.add(-99)
s.topk()
s.add(4)
s.topk()
s.add(100)
s.topk()
		
输出: 
[10, 3]
[1000, 10, 3]
[1000, 10, 4]
[1000, 100, 10]

解释:
s = new Solution(3);
>> 生成了一个新的数据结构, 并且 k = 3.
s.add(3)
s.add(10)
s.topk()
>> 返回 [10, 3]
s.add(1000)
s.add(-99)
s.topk()
>> 返回 [1000, 10, 3]
s.add(4)
s.topk()
>> 返回 [1000, 10, 4]
s.add(100)
s.topk()
>> 返回 [1000, 100, 10]
```

**样例2**

```
输入: 
s = new Solution(1);
s.add(3)
s.add(10)
s.topk()
s.topk()

输出: 
[10]
[10]

解释:
s = new Solution(1);
>> 生成了一个新的数据结构, 并且 k = 1.
s.add(3)
s.add(10)
s.topk()
>> 返回 [10]
s.topk()
>> 返回 [10]
```



```python
import heapq
class Solution:

    def __init__(self, k):
        self.k = k
        self.heap = []
        
    def add(self, num):
        heapq.heappush(self.heap, num)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)

    def topk(self):
        return sorted(self.heap, reverse=True)
```
pass