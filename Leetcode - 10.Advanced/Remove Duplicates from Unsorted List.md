

**样例 1：**
```
输入：1->2->1->3->3->5->6->3->null
输出：1->2->3->5->6->null
```
**样例 2：**
```
输入：2->2->2->2->2->null
输出：2->null
```


```python
class Solution:
    # @param head, a ListNode
    # @return a ListNode
    def remove_duplicates(self, head):
        # Write your code here
        mp = dict()
        if head is None:
            return head;
        mp[head.val] = 1
        tail = head;
        now = head.next;
        while now is not None:
            if now.val not in mp:
                mp[now.val] = 1
                tail.next = now
                tail = tail.next
            now = now.next;

        tail.next = None;
        return head;
```
pass