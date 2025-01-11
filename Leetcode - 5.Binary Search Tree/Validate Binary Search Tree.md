
**样例 1：**
输入：
```
tree = {-1}
```
输出：
```
true
```
解释：
```
二叉树如下(仅有一个节点）:
        -1
这是二叉查找树。
```

**样例 2：**
输入：
```
tree = {2,1,4,#,#,3,5}
```
输出：
```
true
```
解释：
```
        二叉树如下：
          2
         / \
        1   4
           / \
          3   5
这是二叉查找树。
```



```python
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    def is_valid_b_s_t(self, root: TreeNode) -> bool:
        stack, inorder = [], float('-inf')
        
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if root.val <= inorder:
                return False
            inorder = root.val
            root = root.right

        return True
```
pass