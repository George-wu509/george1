
**样例 1:**
```
输入: 7->1->6->null, 5->9->2->null
输出: 2->1->9->null	
样例解释: 617 + 295 = 912, 912 转换成链表:  2->1->9->null
```
**样例 2:**
```
输入:  3->1->5->null, 5->9->2->null
输出: 8->0->8->null	
样例解释: 513 + 295 = 808, 808 转换成链表: 8->0->8->null
```


```python
def add_lists(self, l1, l2) -> list:       
	dummy = ListNode(None)
	tail = dummy
	
	carry = 0 
	while l1 or l2 or carry:
		num = 0 
		if l1:
			num += l1.val 
			l1 = l1.next
		if l2:
			num += l2.val 
			l2 = l2.next
		num += carry
		digit, carry = num % 10, num // 10
		node = ListNode(digit)
		tail.next, tail = node, node 
	return dummy.next
```
pass