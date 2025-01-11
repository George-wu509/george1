

**样例 1：**
输入：
```
Tree = {5,3,6,2,4}
value = 3
```
输出：
```
{5,2,6,#,4} 或 {5,4,6,2}
```
解释：
```
给定了以下二叉搜索树:
    5
   / \
  3   6
 / \
2   4
移去3，你可以返回:
    5
   / \
  2   6
   \
    4
或
    5
   / \
  4   6
 /
2
```

**样例 2：**
输入：
```
Tree = {5,3,6,2,4}
value = 4
```
输出：
```
{5,3,6,2}
```
解释：
```
给定了以下二叉搜索树:
    5
   / \
  3   6
 / \
2   4
移去4，你应该返回
    5
   / \
  3   6
 /
2
```


```python
class Solution:
    """
    @param root: The root of the binary search tree.
    @param value: Remove the node with given value.
    @return: The root of the binary search tree after removal.
    """    
    def __init__(self):
        self.ans = []

    def inorder(self, root, value):
        if root is None:
            return

        self.inorder(root.left, value)
        if root.val != value:
            self.ans.append(root.val)
        self.inorder(root.right, value)
    
    def build(self, l, r):
        if l == r:
            node = TreeNode(self.ans[l])
            return node

        if l > r:
            return None

        mid = (l+r) // 2
        node = TreeNode(self.ans[mid])
        node.left = self.build(l, mid-1)
        node.right = self.build(mid+1, r)
        return node
    def removeNode(self, root, value):
        # write your code here
        self.inorder(root, value)
        return self.build(0, len(self.ans)-1)
```
pass