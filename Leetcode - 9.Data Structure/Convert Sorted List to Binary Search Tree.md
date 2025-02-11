Lintcode 106
给出一个所有元素以升序排序的单链表，将它转换成一棵高度平衡的二叉搜索树

**样例 1：**
输入：
```python
"""
linked list = 1->2->3
```
输出：
```python
"""
  2  
 / \
1   3
```
解释：
将链表转换成一棵高度平衡的二叉搜索树。

**样例 2：**
输入：
```python
"""
linked list = 2->3->6->7
```
输出：
```python
"""
   3
  / \
 2   6
      \
       7
```
解释：
可能有多个答案，您可以返回任何一个。


```python
class Solution:
    """
    @param: head: The first node of linked list.
    @return: a tree node
    """
    def sorted_list_to_b_s_t(self, head):
        length = self.get_linked_list_length(head)
        root, next = self.convert(head, length)
        return root
    
    def get_linked_list_length(self, head):
        length = 0
        while head:
            length += 1
            head = head.next
        return length
        
    def convert(self, head, length):
        if length == 0:
            return None, head
        
        left_root, middle = self.convert(head, length // 2)
        right_root, next = self.convert(middle.next, length - length // 2 - 1)
        
        root = TreeNode(middle.val)
        root.left = left_root
        root.right = right_root
        
        return root, next
```
pass