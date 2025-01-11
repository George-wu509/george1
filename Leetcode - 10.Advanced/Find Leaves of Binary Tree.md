
**样例1**  
输入： {1,2,3,4,5}  
输出： `[[4, 5, 3], [2], [1]]`.  
解释：
```
    1
   / \
  2   3
 / \     
4   5    
```

**样例2**  
输入： {1,2,3,4}  
输出： `[[4, 3], [2], [1]]`.  
解释：
```
    1
   / \
  2   3
 /
4 
```



```python
class Solution:
    """
    @param: root: the root of binary tree
    @return: collect and remove all leaves
    """
    def __init__(self):
        self.leaves = []
    def findLeaves(self, root):
        # write your code here
        self.tree_height(root)
        return self.leaves
    
    def tree_height(self, root):
        if root == None:
            return -1
        left_height = self.tree_height(root.left)
        right_height = self.tree_height(root.right)
        height = 1 + max(left_height, right_height)
        if height >= len(self.leaves):
            self.leaves.append([])
        self.leaves[height].append(root.val)
        return height
```
pass