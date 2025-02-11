Lintcode 152
给定两个整数 `n` 和 `k`. 返回从 `1, 2, ... , n` 中选出 `k` 个数的所有可能的组合.


**样例 1:**
```python
"""
输入: n = 4, k = 2
输出: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
```
**样例 2:**
```python
"""
输入: n = 4, k = 1
输出: [[1],[2],[3],[4]]
```


```python
class Solution:
    def combine(self, n, k):
        combinations = []
        self.helper(1, [], n, k, combinations)
        return combinations
    
    # 递归地求解问题
    # pos代表已访问到哪个数，combination代表当前地组合
    def helper(self, pos, combination, n, k, combinations):
        # 第一个递归出口，如果满足 k 个，将这种组合加入最终结果中
        if len(combination) == k:
            combinations.append(combination[:])
            return
        
        # 第二个递归出口，如果以访问完1 ~ n所有数字，退出当前函数
        if pos == n + 1:
            return
        
        # 可行性剪枝，如果后面的数都放入，个数不足k的话，退出
        if len(combination) + n - pos + 1 < k:
            return
        
        # 如果将当前数放入组合，将pos加入combination，继续递归
        combination.append(pos)
        self.helper(pos + 1, combination, n, k, combinations)
        # 递归结束后，弹出pos，还原状态
        combination.pop()
        
        # 不将pos加入combination的情况
        self.helper(pos + 1, combination, n, k, combinations)
```
pass