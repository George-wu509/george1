
例1:
```
输入:
nums = [1,3,1,4,4,2]
输出:
[1,3,4,2,?,?]
4

解释:
1. 将重复的整数移动到 nums 的尾部 => nums = [1,3,4,2,?,?].
2. 返回 nums 中唯一整数的数量  => 4.
事实上我们并不关心你把什么放在了 ? 处, 只关心没有重复整数的部分.
```
例2:
```
输入:
nums = [1,2,3]
输出:
[1,2,3]
3
```


```python
class Solution:
    def deduplication(self, nums):
        d, result = {}, 0
        for num in nums:
            if num not in d:
                d[num] = True
                nums[result] = num
                result += 1

        return result
```
pass