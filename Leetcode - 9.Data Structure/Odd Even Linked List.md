Lintcode 1292
给定单链表，将所有奇数节点连接在一起，然后将偶数节点连接在一起。 请注意，这里我们讨论的是节点编号，而不是节点中的值

**样例1:**
```python
"""
输入： 1->2->3->4->5->NULL
输出： 1->3->5->2->4->NULL
```
**样例2:**

```python
"""
输入： 2->1->null
输出： 2->1->null
```


```python
def odd_even_list(self, head):
	if head is None:
		return head
	odd = head
	evenHead = head.next
	even = evenHead
	while even and even.next:
		odd.next = even.next
		odd = odd.next
		even.next = odd.next
		even = even.next
	odd.next = evenHead
	return head
```
pass