
406
给定一个由 `n` 个正整数组成的数组和一个正整数 `s` ，请找出该数组中满足其和 ≥ s 的最小长度子数组。如果无解，则返回 -1。

**样例 1:**
```python
输入: [2,3,1,2,4,3], s = 7
输出: 2
解释: 子数组 [4,3] 是该条件下的最小长度子数组。
```
**样例 2:**
```python
输入: [1, 2, 3, 4, 5], s = 100
输出: -1
```

```python
    def minimum_Size(self, nums, s):
        if nums is None or len(nums) == 0:
            return -1

        n = len(nums)
        minLength = n + 1
        sum = 0
        j = 0
        for i in range(n):
            while j < n and sum < s:
                sum += nums[j]
                j += 1
            if sum >= s:
                minLength = min(minLength, j - i)

            sum -= nums[i]
            
        if minLength == n + 1:
            return -1
            
        return minLength
```
pass

解釋:
1. 第一個指針i從左(id=0)一步步移到右. 從一開始i在0的位置, 第二個指針j 也從id=0開始慢慢往右直到並計算sum直到sum>s. 紀錄長度=minLength
2. 第一個指針i往右移一步(id=1), sum減掉= id=0的值, 然後第二個指針j 從剛剛j停留的位置繼續往右移直到sum>s. 紀錄長度=minLength = min(minLength, j-i)
3. 繼續直到第一個指針i到最右端, minLength就是最小長度