
**样例 1：**
输入：
```
list = null
x = 0
```
输出：
```
null
```
解释：
空链表本身满足要求

**样例 2：**
输入：
```
list = 1->4->3->2->5->2->null
x = 3
```
输出：
```
1->2->2->4->3->5->null
```
解释：
要保持原有的相对顺序。


```python
def partition(self, head, x):
	if head is None:
		return head
	aHead, bHead = ListNode(0), ListNode(0)
	aTail, bTail = aHead, bHead
	while head is not None:
		if head.val < x:
			aTail.next = head
			aTail = aTail.next
		else:
			bTail.next = head
			bTail = bTail.next
		head = head.next
	bTail.next = None
	aTail.next = bHead.next
	return aHead.next
```
pass