Lintcode 521
给一个整数数组 `nums`，在逻辑上去除重复的元素，返回去除后的数组长度 `n`，使得通过去重操作数组 `nums` 的前 `n` 个元素中，包含原数组 `nums` 去重后的所有元素。

你应该做这些事

1.在原数组上操作  
2.将去除重复之后的元素放在数组的开头  
3.返回去除重复元素之后的元素个数
例1:
```python
"""
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
```python
"""
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