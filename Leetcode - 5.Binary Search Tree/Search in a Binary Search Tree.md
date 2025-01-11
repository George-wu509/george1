
**样例 1:**
```
输入: value = 2
        4
       / \
      2   7
     / \
    1   3
输出: 节点 2
```
**样例 2:**
```
输入: value = 5
        4
       / \
      2   7
     / \
    1   3
输出: null
```


```python
class Solution:

    def search_b_s_t(self, root: TreeNode, val: int) -> TreeNode:
        while root:
            if root.val == val:
                return root
            root = root.left if root.val > val else root.right
        
        return None
```
pass