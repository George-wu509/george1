Lintcode 1320
给定一个整数数组，查找数组是否包含任何重复项。 如果数组中某个值至少出现两次，则函数应返回true，如果每个元素都是不同的，则返回false。

**样例 1：**
```python
"""
输入：nums = [1, 1]
输出：True
```
**样例 2：**
```python
"""
输入：nums = [1, 2, 3]
输出：False
```

```python
"""
    def contains_duplicate(self, nums: List[int]) -> bool:
        s = set()
        for x in nums:
            if x in s: return True
            s.add(x)
        return False
```
pass