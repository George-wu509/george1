
**样例1**
```
输入：
{8,3,10,1,6,#,14,#,#,4,7,13}
5
13
输出： {8, 6, 10, #, 7, #, 13}
说明：树的图片在题面描述里已经给出
```
**样例2**
```
输入：
{1,0,2}
1
2
输出： {1,#,2}
说明:
输入是
  1
 / \
0   2
输出是
  1
   \
    2
```



```python
class Solution:
    """
    @param root: given BST
    @param minimum: the lower limit
    @param maximum: the upper limit
    @return: the root of the new tree 
    """
    def trim_b_s_t(self, root, minimum, maximum):
        dummy = TreeNode(float('inf'))
        dummy.left = root
        
        self.trim_min(dummy, dummy.left, minimum)
        self.trim_max(dummy, dummy.left, maximum)
        return dummy.left

    def trim_min(self, parent, node, minimum):
        if node is None:
            return
        if node.val >= minimum:
            self.trim_min(node, node.left, minimum)
            return
        if parent.left == node:
            parent.left = node.right
        else:
            parent.right = node.right
        self.trim_min(parent, node.right, minimum)
        
    def trim_max(self, parent, node, maximum):
        if node is None:
            return
        if node.val <= maximum:
            self.trim_max(node, node.right, maximum)
            return
        if parent.left == node:
            parent.left = node.left
        else:
            parent.right = node.left
        self.trim_max(parent, node.left, maximum)
```
pass