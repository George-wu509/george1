
**样例 1：**
输入：
```
数组 = [1, 1, 1, 1, 2, 2, 2]
```
输出：
```
1
```
解释：
数组1的个数大于数组元素的二分之一。

**样例 2：**
输入：
```
数组 = [1, 1, 1, 2, 2, 2, 2]
```
输出：
```
2
```
解释：
数组中2的个数大于数组元素的二分之一。




```python
class Solution:
    def majority_number(self, nums: List[int]) -> int:
        n = len(nums)
        x, cnt = -1, 0
        for i in nums:
            if not cnt:
                x = i
            if x == i:
                cnt += 1
            else:
                cnt -= 1 
        return x if cnt and nums.count(x) > n // 2 else -1
```
pass