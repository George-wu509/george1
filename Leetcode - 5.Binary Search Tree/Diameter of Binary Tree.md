

**样例 1:**
```
给定一棵二叉树 
          1
         / \
        2   3
       / \     
      4   5    
返回3, 这是路径[4,2,1,3] 或者 [5,2,1,3]的长度.
```
**样例 2:**
```
输入:[2,3,#,1]
输出:2

解释:
      2
    /
   3
 /
1
```


```python
class Solution:

    max_len = 0

    def diameter_of_binary_tree(self, root: TreeNode) -> int:

        self.dfs(root)

        return self.max_len

    def dfs(self, node):

        if not node:

            return 0

        left_depth  = self.dfs(node.left)

        right_depth = self.dfs(node.right)

        self.max_len = max(self.max_len, left_depth + right_depth)

        return max(left_depth, right_depth) + 1
```
pass