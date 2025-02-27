Lintcode 649
给定一个二叉树，其中所有右节点要么是具有兄弟节点的叶节点(有一个共享相同父节点的左节点)或空白，将其倒置并将其转换为树，其中原来的右节点变为左叶子节点。返回新的根节点。


```python
"""
输入: {1,2,3,4,5}
输出: {4,5,2,#,#,3,1}
说明:
输入是
    1
   / \
  2   3
 / \
4   5
输出是
   4
  / \
 5   2
    / \
   3   1
```
**样例2**
```python
"""
输入: {1,2,3,4}
输出: {4,#,2,3,1}
说明:
输入是
    1
   / \
  2   3
 /
4
输出是
   4
    \
     2
    / \
   3   1
```


```python
class Solution:
    def upside_down_binary_tree(self, root):
        if root is None:
            return None
        return self.dfs(root)
        
    def dfs(self, root):
        if root.left is None:
            return root
        newRoot = self.dfs(root.left)
        root.left.right = root
        root.left.left = root.right
        root.left = None
        root.right = None
        
        return newRoot
```
pass