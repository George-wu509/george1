
**样例 1:**
```
输入:
{1,-5,2,1,2,-4,-5}
输出:1
说明
这棵树如下所示：
     1
   /   \
 -5     2
 / \   /  \
1   2 -4  -5 
整颗树的和是最小的，所以返回根节点1.
```
**样例 2:**
```
输入:
{1}
输出:1
说明:
这棵树如下所示：
   1
这棵树只有整体这一个子树，所以返回1.
```


```python
class Solution:
    weight = float('Inf')
    node = None

    def find_subtree(self, root: TreeNode) -> TreeNode:
        self.dfs(root)
        return self.node      
    
    def dfs(self, node):
        if not node:
            return 0
        sum_L = self.dfs(node.left)
        sum_R = self.dfs(node.right)
        sum_all = sum_L + sum_R + node.val
        if sum_all < self.weight:
            self.weight = sum_all
            self.node = node
        return sum_all
```
pass