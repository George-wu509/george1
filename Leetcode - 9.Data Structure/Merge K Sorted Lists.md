

**样例 1：**
输入：
```
lists = [2->4->null,null,-1->null]
```
输出：
```
-1->2->4->null
```
解释：
将2->4->null、null和-1->null合并成一个升序的链表。

**样例 2：**
输入：
```
lists = [2->6->null,5->null,7->null]
```
输出：
```
2->5->6->7->null
```
解释：
将2->6->null、5->null和7->null合并成一个升序的链表。

```python
class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        if not lists:
            return None
        
        return self.merge_range_lists(lists, 0, len(lists) - 1)
        
    def merge_range_lists(self, lists, start, end):
        if start == end:
            return lists[start]
        
        mid = (start + end) // 2
        left = self.merge_range_lists(lists, start, mid)
        right = self.merge_range_lists(lists, mid + 1, end)
        return self.merge_two_lists(left, right)
        
    def merge_two_lists(self, head1, head2):
        tail = dummy = ListNode(0)
        while head1 and head2:
            if head1.val < head2.val:
                tail.next = head1
                head1 = head1.next
            else:
                tail.next = head2
                head2 = head2.next
            tail = tail.next
            
        if head1:
            tail.next = head1
        if head2:
            tail.next = head2
        
        return dummy.next
```
pass