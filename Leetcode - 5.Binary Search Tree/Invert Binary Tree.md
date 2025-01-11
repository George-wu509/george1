

**样例 1:**
```
输入: {1,3,#}
输出: {1,#,3}
解释:
	  1    1
	 /  =>  \
	3        3
```
**样例 2:**
```
输入: {1,2,3,#,#,4}
输出: {1,3,2,#,4}
解释: 
	
      1         1
     / \       / \
    2   3  => 3   2
       /       \
      4         4
```


```python
class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def invert_binary_tree(self, root: TreeNode):
        self.invertTree(root)

    def invertTree(self, node):
        if node is None:
            return None

        node.left, node.right = node.right, node.left
        self.invertTree(node.left)
        self.invertTree(node.right)
```
pass