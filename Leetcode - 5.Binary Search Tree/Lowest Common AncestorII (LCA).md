

**样例 1:**
```
输入：{4,3,7,#,#,5,6},3,5
输出：4
解释：
     4
     / \
    3   7
       / \
      5   6
LCA(3, 5) = 4
```
**样例 2:**
```
输入：{4,3,7,#,#,5,6},5,6
输出：7
解释：
      4
     / \
    3   7
       / \
      5   6
LCA(5, 6) = 7
```


```python
class Solution:

    def lowestCommonAncestorII(self, root, A, B):
        if not root:
            return None
        parent_set = set()
        curr = A
        while curr is not None:
            parent_set.add(curr)
            curr = curr.parent
        
        curr = B
        while curr is not None:
            if curr in parent_set:
                return curr
            curr = curr.parent
        return None
```
pass