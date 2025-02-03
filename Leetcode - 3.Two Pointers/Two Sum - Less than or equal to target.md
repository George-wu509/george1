609
给定一个整数数组，找出这个数组中有多少个不同的下标对，满足下标对中的两个下标所对应元素之和小于或等于目标值。返回下标对数。

样例
例1:
```python
输入: nums = [2, 7, 11, 15], target = 24. 
输出: 5. 
解释:
2 + 7 < 24
2 + 11 < 24
2 + 15 < 24
7 + 11 < 24
7 + 15 < 24
```
例2:
```python
输入: nums = [1], target = 1. 
输出: 0. 
```


```python
class Solution:
    # @param nums {int[]} an array of integer
    # @param target {int} an integer
    # @return {int} an integer
    def twoSum5(self, nums, target):
        # Write your code here
        l, r = 0, len(nums)-1
        cnt = 0
        nums.sort()
        while l < r:
            value = nums[l] + nums[r]
            if value > target:
                r -= 1
            else:
                cnt += r - l
                l += 1
        return cnt
```
pass

解釋為何用cnt+=r-l:
step1:  [ l, r ] = [ 0, 3 ]  value=17 ---> cnt+=3 代表 [ 0, 3 ], [ 0, 2 ], [ 0, 1 ]
step2:  [ l, r ] = [ 1, 3 ]  value=22 ---> cnt+=3 代表 [ 1, 3 ], [ 1, 2 ]
