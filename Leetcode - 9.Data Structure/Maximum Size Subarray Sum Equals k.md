Lintcode 911
给一个数组`nums`和目标值`k`，找到数组中最长的子数组，使其中的元素和为k。如果没有，则返回0。

**样例1**
```python
"""
输入: nums = [1, -1, 5, -2, 3], k = 3
输出: 4
解释:
子数组[1, -1, 5, -2]的和为3，且长度最大
```
**样例2**
```python
"""
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

nums =  [-2, -1, 2, 1]

解釋:
step1: 用dict建立一個hash table m, 另一個變數sum是前缀和 list.
step2: 從id=0往右, 每個id計算前缀和 sum, 並把缀和加入m 
step3: 每一個step計算新的prefix_sum, 並check是否在hash table裡有跟當前prefix_sum一樣的值
因為這代表新的id的prefix_sum(new)到以前存的prefix_sum(old)
prefix_sum(new)-prefix_sum(old)=0