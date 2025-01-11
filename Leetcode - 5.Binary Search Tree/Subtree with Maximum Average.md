
**样例1**
```
输入：
{1,-5,11,1,2,4,-2}
输出：11
说明:
这棵树如下所示：
     1
   /   \
 -5     11
 / \   /  \
1   2 4    -2 
11子树的平均值是4.333，为最大的。
```
**样例2**
```
输入：
{1,-5,11}
输出：11
说明:
     1
   /   \
 -5     11
1,-5,11 三棵子树的平均值分别是 2.333,-5,11。因此11才是最大的
```


```python
class Solution:
    ans_average, ans_node = float('-inf'), None 
    def find_subtree2(self, root: TreeNode) -> TreeNode:
        self.dfs(root)
        return self.ans_node
        
    def dfs(self, node):
        if not node:
            return 0, 0
        left_sum, left_n = self.dfs(node.left)
        right_sum, right_n = self.dfs(node.right)

        total_n = left_n + right_n + 1
        total_sum = left_sum + right_sum + node.val
        if total_sum / total_n > self.ans_average:
            self.ans_average = total_sum / total_n
            self.ans_node = node

        return total_sum, total_n
```
pass