Lintcode 453
将一棵二叉树按照前序遍历拆解成为一个 `假链表`。所谓的假链表是说，用二叉树的 _right_ 指针，来表示链表中的 _next_ 指针

**样例 1：**
```python
"""
输入：{1,2,5,3,4,#,6}
输出：{1,#,2,#,3,#,4,#,5,#,6}
解释：
     1
    / \
   2   5
  / \   \
 3   4   6
 
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
```
**样例 2：**
```python
"""
输入：{1}
输出：{1}
解释：
         1
         1
```


```python
class Solution:

    def flatten(self, root):
        self.flatten_and_return_last_node(root)
        
    # restructure and return last node in preorder
    def flatten_and_return_last_node(self, root):
        if root is None:
            return None
            
        left_last = self.flatten_and_return_last_node(root.left)
        right_last = self.flatten_and_return_last_node(root.right)
        
        # connect 
        if left_last is not None:
            left_last.right = root.right
            root.right = root.left
            root.left = None
            
        return right_last or left_last or root
```
pass