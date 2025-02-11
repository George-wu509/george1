lintcode 165
将两个排序（升序）链表合并为一个新的升序排序链表

```python
"""
样例 1:
	输入: list1 = null, list2 = 0->3->3->null
	输出: 0->3->3->null


样例2:
	输入:  list1 =  1->3->8->11->15->null, list2 = 2->null
	输出: 1->2->3->8->11->15->null
```


```python
def merge_two_lists(self, l1, l2):
	dummy = ListNode(0)
	tmp = dummy
	while l1 != None and l2 != None:
		if l1.val < l2.val:
			tmp.next = l1
			l1 = l1.next
		else:
			tmp.next = l2
			l2 = l2.next
		tmp = tmp.next
	if l1 != None:
		tmp.next = l1
	else:
		tmp.next = l2
	return dummy.next
```
pass