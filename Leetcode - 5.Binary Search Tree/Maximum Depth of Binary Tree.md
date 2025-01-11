
**样例 1：**
输入：
```
tree = {}
```
输出：
```
0
```
解释：
空树的深度是0。

**样例 2：**
输入：
```
tree = {1,2,3,#,#,4,5}
```
输出：
```
3
```
解释：
树表示如下，深度是3  
1  
/ \  
2   3  
/  \  
4    5  
它将被序列化为{1,2,3,#,#,4,5}


```python
class Solution:

    def max_depth(self, root: TreeNode) -> int:
        return self.dfs(root)

    def dfs(self, node):
        if not node:
            return 0
        left_depth = self.dfs(node.left)
        right_depth = self.dfs(node.right)
        return max(left_depth, right_depth) + 1
```
pass