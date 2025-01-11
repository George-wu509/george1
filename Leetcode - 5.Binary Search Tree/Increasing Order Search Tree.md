

**样例 1:**
```
输入：root = {5,3,6,2,4,#,8,1,#,#,#,7,9}

       5
      / \
    3    6
   / \    \
  2   4    8
 /        / \ 
1        7   9

输出：{1,#,2,#,3,#,4,#,5,#,6,#,7,#,8,#,9}
解释：
 1
  \
   2
    \
     3
      \
       4
        \
         5
          \
           6
            \
             7
              \
               8
                \
                 9  
```
**样例 2:**

```
输入: root = {8,3,10,1,6,#,14,#,#,4,7,13,#}
       8
      /  \
    3     10
   / \      \
  1   6      14
      / \   / 
     4   7  13
输出: {1,#,3,#,4,#,6,#,7,#,8,#,10,#,13,#,14}
解释：
1
 \
  3
   \
    4
     \
      6
       \
        7
         \
          8
           \
            10
             \
              13
               \
                14
```


```python
class Solution:
    def inorder(self, root):
        stack = []
        res = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            res.append(root.val)
            root = root.right
        return res

    def increasing_b_s_t(self, root: TreeNode) -> TreeNode:
        node_list = self.inorder(root)
        dummy = TreeNode(-1)
        node = dummy
        for value in node_list:
            node.right = TreeNode(value)
            node = node.right
            
        return dummy.right
```
pass