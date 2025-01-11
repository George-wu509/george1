
**样例1**
```
输入: 
{4, 3, 7, #, #, 5, 6}
3 5
5 6
6 7 
5 8
输出: 
4
7
7
null
解释:
  4
 / \
3   7
   / \
  5   6

LCA(3, 5) = 4
LCA(5, 6) = 7
LCA(6, 7) = 7
LCA(5, 8) = null
```

**样例2**
```
输入:
{1}
1 1
输出: 
1
说明：
这棵树只有一个值为1的节点
```


```python
class Solution:
    """
    @param: root: The root of the binary tree.
    @param: A: A TreeNode
    @param: B: A TreeNode
    @return: Return the LCA of the two nodes.
    """
    def lowestCommonAncestor3(self, root, A, B):
        A_exist, B_exist, lca = self.getLCA(root, A, B)
        #if A_exist and B_exist:
        #    return lca
        #else:
        #    return None
        return lca if A_exist and B_exist else None

    def getLCA(self, node, A, B):
        if node is None:
            return False, False, None
        
        Aleft_exist, Bleft_exist, left_node = self.getLCA(node.left, A, B)
        Aright_exist, Bright_exist, right_node = self.getLCA(node.right, A, B)

        A_exist = Aleft_exist or Aright_exist or node == A
        B_exist = Bleft_exist or Bright_exist or node == B

        if node == A or node == B:
            return A_exist, B_exist, node

        if left_node is not None and right_node is not None:
            return A_exist, B_exist, node

        if left_node is not None:
            return A_exist, B_exist, left_node

        if right_node is not None:
            return A_exist, B_exist, right_node

        return A_exist, B_exist, None
```
pass