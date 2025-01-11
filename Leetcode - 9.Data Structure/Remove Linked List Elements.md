
**样例 1：**
```
输入：head = 1->2->3->3->4->5->3->null, val = 3
输出：1->2->4->5->null
```
**样例 2：**
```
输入：head = 1->1->null, val = 1
输出：null
```


```python
def remove_elements(self, head, val):
	if head == None:
		return head
	dummy = ListNode(0)
	dummy.next = head
	pre = dummy
	while head:
		if head.val == val:
			pre.next = head.next
			head = pre
		pre = head
		head = head.next
	return dummy.next  
```
pass