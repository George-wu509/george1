

**样例 1：**
输入：
```
tree = {2}
```
输出：
```
2
```
解释：
只有一个节点2  

**样例 2：**
输入：
```
tree = {1,2,3}
```
输出：
```
6
```
解释：
```
如下图，最长路径为2-1-3
      1
     / \
    2   3
```

**样例 3：**  
输入：
```
tree = {1, 2, 3, 4, 9, 6, #, 1, 3, 4, #, 8, 12, #, 14, #, 3, 6}
```
输出：
```
43
```
解释：
```
如下图，最长路径为 14-1-4-2-1-3-6-12
```

![94_1.png](https://media-cn.lintcode.com/new_storage_v2/public/202404/7c49ac1733374d06b36db7ffd5cc9812/94_1.png)

相关知识


```python
class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def max_path_sum(self, root):
        maxSum, _ = self.maxPathHelper(root)
        return maxSum
        
    def maxPathHelper(self, root):
        if root is None:
            return -sys.maxsize, 0
        
        left = self.maxPathHelper(root.left)
        right = self.maxPathHelper(root.right)
        maxpath = max(left[0], right[0], root.val + left[1] + right[1])
        single = max(left[1] + root.val, right[1] + root.val, 0)
        
        return maxpath, single
```
pass