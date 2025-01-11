

**样例 1:**
```
输入: {5,10,10,#,#,2,3}
输出: true
解释: 
  原始的树:
     5
    / \
   10 10
     /  \
    2    3
  两棵子树:
     5       10
    /       /  \
   10      2    3
```
**样例 2:**
```
输入: {1,2,10,#,#,2,20}
输出: false
解释: 
  原始的树:
     1
    / \
   2  10
     /  \
    2    20
```


```python
class Solution:
    """
    @param root: a TreeNode
    @return: return a boolean
    """

    def check_equal_tree(self, root):
        self.mp = {}
        sum = self.dfs(root)
        if(sum == 0):
            return self.mp[0] > 1
        return sum % 2 == 0 and not self.mp.get(sum / 2) == None

    def dfs(self, root):
        if(root == None):
            return 0
        sum = root.val + self.dfs(root.left) + self.dfs(root.right)
        if(self.mp.get(sum) == None):
            self.mp[sum] = 1
        else:
            self.mp[sum] += 1
        return sum
```
pass