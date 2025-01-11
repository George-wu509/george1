
```
示例 1:

输入: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
输出: [1,2],[1,4],[1,6]
解释: 返回序列中的前 3 对数：
     [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]
```

```
示例 2:

输入: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
输出: [1,1],[1,1]
解释: 返回序列中的前 2 对数：
     [1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]
```

```
示例 3:

输入: nums1 = [1,2], nums2 = [3], k = 3 
输出: [1,3],[2,3]
解释: 也可能序列中所有的数对都被返回:[1,3],[2,3]
```


```python
    def k_smallest_pairs(self, nums1, nums2, k):
        # write your code here
        n1, n2 = len(nums1), len(nums2)
        import heapq
        heap = [[nums1[0] + nums2[0], 0, 0]]
        seen = set((0, 0))
        res = []
        for i in range(min(k, n1 * n2)):
            _, idx1, idx2 = heapq.heappop(heap)
            res.append([nums1[idx1], nums2[idx2]])
            if idx1 < n1 - 1 and (idx1 + 1, idx2) not in seen:
                heapq.heappush(heap, [nums1[idx1 + 1] + nums2[idx2], idx1 + 1, idx2])
                seen.add((idx1 + 1, idx2))
            if idx2 < n2 - 1 and (idx1, idx2 + 1) not in seen:
                heapq.heappush(heap, [nums1[idx1] + nums2[idx2 + 1], idx1, idx2 + 1])
                seen.add((idx1, idx2 + 1))
        return res
```
pass