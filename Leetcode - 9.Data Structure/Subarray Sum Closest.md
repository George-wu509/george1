

**样例1**
```
输入: 
[-3,1,1,-3,5] 
输出: 
[0,2]
解释: 返回 [0,2], [1,3], [1,1], [2,2], [0,4] 中的任意一个均可。
```
**样例2**
```
输入: 
[2147483647]
输出: 
[0,0]
```

```python
    def subarray_sum_closest(self, nums):
        prefix_sum = [(0, -1)]
        for i, num in enumerate(nums):
            prefix_sum.append((prefix_sum[-1][0] + num, i))
            
        prefix_sum.sort()
        
        closest, answer = sys.maxsize, []
        for i in range(1, len(prefix_sum)):
            if closest > prefix_sum[i][0] - prefix_sum[i - 1][0]:
                closest = prefix_sum[i][0] - prefix_sum[i - 1][0]
                left = min(prefix_sum[i - 1][1], prefix_sum[i][1]) + 1
                right = max(prefix_sum[i - 1][1], prefix_sum[i][1])
                answer = [left, right]
        
        return answer
```
pass