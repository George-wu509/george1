
**样例1**
```
输入: nums = [1, -1, 5, -2, 3], k = 3
输出: 4
解释:
子数组[1, -1, 5, -2]的和为3，且长度最大
```
**样例2**
```
输入: nums = [-2, -1, 2, 1], k = 1
输出: 2
解释:
子数组[-1, 2]的和为1，且长度最大
```


```python
    def max_sub_array_len(self, nums, k):
        m = {}
        ans = 0
        m[k] = 0
        n = len(nums)
        sum = [0 for i in range(n + 1)]
        for i in range(1, n + 1):
            sum[i] = sum[i - 1] + nums[i - 1]
            if sum[i] in m:
                ans = max(ans, i - m[sum[i]])
            if sum[i] + k not in m:
                m[sum[i] + k] = i
        return ans
```
pass