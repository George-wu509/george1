

**样例1:**
```
输入:
{1,-5,3,1,2,-4,-5}
输出: 3
说明:
这棵树如下所示：
     1
   /   \
 -5     3
 / \   /  \
1   2 -4  -5
```
**样例 2**
```plain
输入:
{10,-5,2,0,3,-4,-5}
输出: 10
说明:
这棵树如下所示：
     10
   /   \
 -5     2
 / \   /  \
0   3 -4  -5 
```


```python
class Solution:
    """
    @param root: the root of tree
    @return: the max node
    """
    def max_node(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        stack = [root]
        maxnode = root
        while stack:
            node = stack.pop(0)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
            if node.val >= maxnode.val:
                maxnode = node
        return maxnode
```
pass