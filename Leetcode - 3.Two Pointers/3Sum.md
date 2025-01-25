57
给出一个有 `n` 个整数的数组 `S`，在 `S` 中找到三个整数 `a`, `b`, `c`，找到所有使得 `a + b + c = 0` 的三元组。

Examples
```python
样例 1：
输入：
numbers = [2,7,11,15]
输出：
[]
解释：
找不到三元组使得三个数和为0。

样例 2：
输入：
numbers = [-1,0,1,2,-1,-4]
输出：
[[-1, 0, 1],[-1, -1, 2]]
解释：
[-1, 0, 1]和[-1, -1, 2]是符合条件的2个三元组。
```

```python
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @return: Find all unique triplets in the array which gives the sum of zero.
    """
    def threeSum(self, nums):
        nums = sorted(nums)
        
        results = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            self.find_two_sum(nums, i + 1, len(nums) - 1, -nums[i], results)
            
        return results

    def find_two_sum(self, nums, left, right, target, results):
        last_pair = None
        while left < right:
            if nums[left] + nums[right] == target:
                if (nums[left], nums[right]) != last_pair:
                    results.append([-target, nums[left], nums[right]])
                last_pair = (nums[left], nums[right])
                right -= 1
                left += 1
            elif nums[left] + nums[right] > target:
                right -= 1
            else:
                left += 1
```
pass