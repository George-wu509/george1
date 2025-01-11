

**样例1**
```plain
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
```plain
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