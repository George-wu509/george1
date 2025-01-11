
**例1:**
```
输入:[9,3,2,4,8],3
输出:4
```
**例2:**
```
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