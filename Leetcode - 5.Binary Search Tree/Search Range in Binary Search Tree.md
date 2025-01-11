
**样例 1：**
输入：
```
tree = {5}
k1 = 6
k2 = 10
```
输出：
```
[]
```
解释：
没有数字介于6和10之间

**样例 2：**
输入：
```
tree = {20,8,22,4,12}
k1 = 10
k2 = 22
```
输出：
```
[12,20,22]
```
解释：
[12,20,22]介于10和22之间


```python
class Solution:
    """
    @param root: param root: The root of the binary search tree
    @param k1: An integer
    @param k2: An integer
    @return: return: Return all keys that k1<=key<=k2 in ascending order
    """
    def search_range(self, root, k1, k2):
        result = []
        self.travel(root, k1, k2, result)
        return result
    
    def travel(self, root, k1, k2, result):
        if root is None:
            return
    	# 剪枝，如果当前节点小于等于k1，不必访问左子树
        if root.val > k1:
            self.travel(root.left, k1, k2, result)
        if k1 <= root.val and root.val <= k2:
            result.append(root.val)
        # 剪枝，如果当前节点大于等于k2，不必访问右子树
        if root.val < k2:
            self.travel(root.right, k1, k2, result)
```
pass