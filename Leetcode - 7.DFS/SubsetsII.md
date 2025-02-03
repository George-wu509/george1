


**样例 1：**
输入：
```python
nums = [0]
```
输出：
```python
[
  [],
  [0]
]
```
解释：
[0]的子集只有[]和[0]。

**样例 2：**
输入：
```python
nums = [1,2,2]
```
输出：
```python
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
```
解释：
[1,2,2]不重复的子集有[],[1],[2],[1,2],[2,2],[1,2,2]。


```python
class Solution:

    def subsets_with_dup(self, nums):
        res = []
        # 排序
        nums.sort()
        # dfs搜索
        self.dfs(nums, 0, [], res)
        return res
        
    def dfs(self, nums, k, subset, res):
        # 当前组合存入res
        res.append(subset[:])
        # 为subset新增一位元素
        for i in range(k, len(nums)):
            # 剪枝
            if (i != k and nums[i] == nums[i - 1]):
                continue
            subset.append(nums[i])
            # 下一层搜索
            self.dfs(nums, i + 1, subset, res)
            # 回溯
            del subset[-1]
```
pass