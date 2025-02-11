Lintcode 606
找到数组中第K大的元素，N远大于K。请注意你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素

**例1:**
```python
"""
输入:[9,3,2,4,8],3
输出:4
```
**例2:**
```python
"""
输入:[1,2,3,4,5,6,8,9,10,7],10
输出:1
```



```python
    def kth_largest_element2(self, nums, k):
        import heapq
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        
        return heapq.heappop(heap)
```
pass