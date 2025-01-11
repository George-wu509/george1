
**样例 1：**
输入：
```
链表 = 1->2->3->4->5->NULL
m = 2 
n = 4
```
输出：
```
1->4->3->2->5->NULL
```
解释：
翻转链表[2,4]位置。

**样例 2：**
输入：
```
链表 = 1->2->3->4->null
m = 2
n = 3
```
输出：
```
1->3->2->4->NULL
```
解释：
翻转链表[2,3]位置。


```python
class Solution:

    def reverse(self, head):
        prev = None
        while head != None:
            next = head.next
            head.next = prev
            prev = head
            head = next
        return prev

    def findkth(self, head, k):
        for i in range(k):
            if head is None:
                return None
            head = head.next
        return head

    def reverse_between(self, head, m, n):
        dummy = ListNode(-1, head)
        mth_prev = self.findkth(dummy, m - 1)
        mth = mth_prev.next
        nth = self.findkth(dummy, n)
        nth_next = nth.next
        nth.next = None

        self.reverse(mth)
        mth_prev.next = nth
        mth.next = nth_next
        return dummy.next
```
pass