
**样例 1:**
```
输入: 6->1->7   2->9->5
输出: 9->1->2
```
**样例 2:**
```
输入: 1->2->3   4->5->6
输出: 5->7->9
```


```python
class Solution:
    """
    @param l1: The first list.
    @param l2: The second list.
    @return: the sum list of l1 and l2.
    """
    # 反转链表
    def reverse(self, l):
        # pre->cur反转为cur->pre,next用于遍历原链表
        pre = None
        cur = l
        next = cur.next
        while next:
            cur.next = pre
            pre = cur
            cur = next
            next = next.next
        cur.next = pre
        return cur
      
    def add_lists2(self, l1, l2):
        l1 = self.reverse(l1)
        l2 = self.reverse(l2)
        ans = ListNode(0)
        cur = ans
        # pre用于最后删去最高位为0的结点
        pre = None
        # l1和l2逐位从低位到高位相加，直到l1或l2到最高位
        while l1 and l2:
            # sum = 进位 + 二者之和
            sum = cur.val + l1.val + l2.val
            cur.val = sum % 10
            cur.next = ListNode(sum // 10)
            l1 = l1.next
            l2 = l2.next
            pre = cur
            cur = cur.next
        # 如果l1 或 l2还有更高位，继续加到答案链表
        while l1:
            sum = cur.val + l1.val
            cur.val = sum % 10
            cur.next = ListNode(sum // 10)
            l1 = l1.next
            pre = cur
            cur = cur.next
        while l2:
            sum = cur.val + l2.val
            cur.val = sum % 10;
            cur.next = ListNode(sum // 10)
            l2 = l2.next
            pre = cur
            cur = cur.next
        if cur.val == 0:
            pre.next = cur.next
        return self.reverse(ans)
```
pass