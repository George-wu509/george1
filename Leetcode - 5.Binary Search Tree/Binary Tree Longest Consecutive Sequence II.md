

**例1:**
```
输入:
{1,2,0,3}
输出:
4
解释:
    1
   / \
  2   0
 /
3
0-1-2-3
```
**例2:**
```
输入:
{3,2,2}
输出:
2
解释:
    3
   / \
  2   2
2-3
```



```python
class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {int} the length of the longest consecutive sequence path
    def longest_consecutive2(self, root):
        # Write your code here
        max_len, _, _, = self.helper(root)
        return max_len
    
    def helper(self, root):
        if root is None:
            return 0, 0, 0

        left_len, left_down, left_up = self.helper(root.left)
        right_len, right_down, right_up = self.helper(root.right)

        down, up = 0, 0
        if root.left is not None and root.left.val + 1 == root.val:
            down = max(down, left_down + 1)

        if root.left is not None and root.left.val - 1 == root.val:
            up = max(up, left_up + 1)

        if root.right is not None and root.right.val + 1 == root.val:
            down = max(down, right_down + 1)

        if root.right is not None and root.right.val - 1 == root.val:
            up = max(up, right_up + 1)

        len = down + 1 + up
        len = max(len, left_len, right_len)

        return len, down, up
```
pass