
例1:
```
输入: [1,7,4,9,2,5]
输出: 6
解释: 整个序列都是一个摆动序列。
```
例2:
```
输入: [1,17,5,10,13,15,10,5,16,8]
输出: 7
解释: 有若干个摆动子序列都满足这个长度。 其中一个为[1,17,10,13,10,16,8]。
```


```python
class Solution:
    def wiggle_max_length(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return n
        
        prevdiff = nums[1] - nums[0]
        ret = (2 if prevdiff != 0 else 1)
        for i in range(2, n):
            diff = nums[i] - nums[i - 1]
            if (diff > 0 and prevdiff <= 0) or (diff < 0 and prevdiff >= 0):
                ret += 1
                prevdiff = diff
        
        return ret
```
pass