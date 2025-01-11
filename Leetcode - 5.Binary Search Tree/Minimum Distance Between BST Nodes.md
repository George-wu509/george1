
**样例 1:**
```
输入: root = {4,2,6,1,3}
输出: 1
解释:
注意，root是树结点对象(TreeNode object)，而不是数组。

给定的树 [4,2,6,1,3,null,null] 可表示为下图:

          4
        /   \
      2      6
     / \    
    1   3  

最小的差值是 1, 它是节点1和节点2的差值, 也是节点3和节点2的差值。
```

**样例 2:**
```
输入: root = {2,1}
输出: 1
解释:
注意，root是树结点对象(TreeNode object)，而不是数组。

给定的树 {2,1} 可表示为下图:

      2
     / 
    1 

最小的差值是 1, 它是节点1和节点2的差值。
```


```python
class Solution:
    """
    @param root:  a Binary Search Tree (BST) with the root node
    @return: the minimum difference
    """
    def min_diff_in_b_s_t(self, root: TreeNode) -> int:
        self.ans = float('inf')
        self.pre = -1
        self.dfs(root)
        return self.ans
    
    def dfs(self, root):
        if not root:
            return None
        self.dfs(root.left)
        if self.pre == -1:
            self.pre = root.val
        else:
            self.ans = min(self.ans, root.val - self.pre)
            self.pre = root.val
        self.dfs(root.right)
```
pass