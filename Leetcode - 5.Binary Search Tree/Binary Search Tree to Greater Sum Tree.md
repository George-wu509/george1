
**样例1:**
```
输入 : {5,2,13}
              5
            /   \
           2     13
输出 : {18,20,13}
             18
            /   \
          20     13
```
**样例2:**
```
输入 : {5,3,15}
              5
            /   \
           3     15
输出 : {20,23,15}
             20
            /   \
          23     15
```


```python
class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {TreeNode} the new root
    def convertBST(self, root):
        # Write your code here
        self.sum = 0
        self.helper(root)
        return root

    def helper(self, root):
        if root is None:
            return
        if root.right:
            self.helper(root.right)
        
        self.sum += root.val
        root.val = self.sum
        if root.left:
            self.helper(root.left)
```
pass