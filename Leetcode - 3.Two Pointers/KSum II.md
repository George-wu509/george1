
样例
**样例 1：**
输入：
```
数组 = [1,2,3,4]
k = 2
target = 5
```
输出：
```
[[1,4],[2,3]]
```
解释：
1+4=5,2+3=5

**样例 2：**
输入：
```
数组 = [1,3,4,6]
k = 3
target = 8
```
输出：
```
[[1,3,4]]
```
解释：
1+3+4=8

```python
class Solution:
    """
    @param: A: an integer array
    @param: k: a postive integer <= length(A)
    @param: targer: an integer
    @return: A list of lists of integer
    """
    def k_sum_i_i(self, nums, k, target):
        nums = sorted(nums)
        subsets = []
        self.dfs(nums, 0, k, target, [], subsets)
        return subsets
        
    def dfs(self, A, index, k, target, subset, subsets):
        if k == 0 and target == 0:
            subsets.append(list(subset))
            return
        
        if k == 0 or target <= 0:
            return
        
        for i in range(index, len(A)):
            subset.append(A[i])
            self.dfs(A, i + 1, k - 1, target - A[i], subset, subsets)
            subset.pop()
```
pass