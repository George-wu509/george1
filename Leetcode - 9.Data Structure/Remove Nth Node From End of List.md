
**样例 1**
输入
```
list = 1->2->3->4->5->null, n = 2
```
输出
```
1->2->3->5->null
```

**样例 2**
输入
```
list = 5->4->3->2->1->null, n = 2
```
输出
```
5->4->3->1->null
```
挑战
假设链表长度是未知的，你会怎么解决呢？



```python
def remove_nth_from_end(self, head, n):
	res = ListNode(0)
	res.next = head
	tmp = res
	for i in range(0, n):
		head = head.next
	while head != None:
		head = head.next
		tmp = tmp.next
	tmp.next = tmp.next.next
	return res.next
```
pass