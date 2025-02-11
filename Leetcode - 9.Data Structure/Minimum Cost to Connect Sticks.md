Lintcode


**样例 1:**
```python
"""
输入：
[2,4,3]
输出：
14
解释：
先将 2 和 3 连接成 5，花费 5；再将 5 和 4 连接成 9；总花费为 14
```
**样例 2:**
```python
"""
输入：
 [1,8,3,5]
输出：
30
```





```python
    def minimum_cost(self, sticks):
        import queue
        if len(sticks) < 2:
            return 0
        minheap = queue.PriorityQueue()
        for num in sticks:
            minheap.put(num)
        res = 0
        while (minheap.qsize() > 1):
            merge = minheap.get() + minheap.get()
            res += merge
            minheap.put(merge)
        return res
```
pass