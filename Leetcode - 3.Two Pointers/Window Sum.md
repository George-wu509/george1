604
给你一个大小为n的整型数组和一个大小为k的滑动窗口，将滑动窗口从头移到尾，输出从开始到结束每一个时刻滑动窗口内的数的和。

**样例 1**
```python
输入：array = [1,2,7,8,5], k = 3
输出：[10,17,20]
解析：
1 + 2 + 7 = 10
2 + 7 + 8 = 17
7 + 8 + 5 = 20
```


```python
    def win_Sum(self, nums, k):
        n = len(nums)
        if n < k or k <= 0:
            return []
        sums = [0] * (n - k + 1)
        for i in range(k):
            sums[0] += nums[i];

        for i in range(1, n - k + 1):
            sums[i] = sums[i - 1] - nums[i - 1] + nums[i + k - 1]

        return sums

```
pass