
**样例 1:**
```
输入: [-3, 1, 2, -3, 4]
输出: [0,2] 或 [1,3]	
样例解释： 返回任意一段和为0的区间即可。
```
**样例 2:**
```
输入: [-3, 1, -4, 2, -3, 4]
输出: [1,5]
```


```python
    def subarray_sum(self, nums):
        prefix_hash = {0: -1}
        prefix_sum = 0
        for i, num in enumerate(nums):
            prefix_sum += num
            if prefix_sum in prefix_hash:
                return prefix_hash[prefix_sum] + 1, i
            prefix_hash[prefix_sum] = i
            
        return -1, -1
```
pass