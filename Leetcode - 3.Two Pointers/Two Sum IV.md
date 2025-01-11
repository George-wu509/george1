Example:
```python
输入：
{4,2,5,1,3}
3
输出： [1,2] (or [2,1])
解释：
二叉搜索树如下：
    4
   / \
  2   5
 / \
1   3
```

```python
输入：
{4,2,5,1,3}
5
输出： [2,3] (or [3,2] or [1,4] or [4,1])
```

```python
import collections
class Solution:
    """
    @param: : the root of tree
    @param: : the target sum
    @return: two numbers from tree which sum is n
    """
    def __init__(self):
        self.s = set()

    def twoSum(self, root, n):
        # write your code here
        if not root:
            return None
        s = set()
        q = collections.deque([root])
        while q:
            node = q.popleft()
            if n - node.val in s:
                return [node.val, n - node.val]
            s.add(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        return None
```
pass