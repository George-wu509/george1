
**样例 1：**
输入：
```
tree = {1,2,3}
```
输出：
```
true
```
解释：
```
如下，是一个平衡的二叉树。
          1  
         / \                
        2   3
```

**样例 2：**
输入：
```
tree = {1,#,2,3,4}
```
输出：
```
false
```
解释：
```
如下，是一个不平衡的二叉树。1的左右子树高度差2
           1  
            \  
            2   
           /  \ 
          3   4
```



```python
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def is_balanced(self, root: TreeNode) -> bool:

        isbalanced, _ = self.GetHeight(root)
        return isbalanced

    def GetHeight(self, Node):
        if not Node:
            return True, 1
        
        left_isbalanced, left_height = self.GetHeight(Node.left)
        right_isbalanced, right_height = self.GetHeight(Node.right)

        Node_height = max(left_height, right_height) + 1
        if not left_isbalanced or not right_isbalanced:
            return False, Node_height
        
        if abs(left_height - right_height) > 1:
            return False, Node_height

        return True, Node_height
```
pass