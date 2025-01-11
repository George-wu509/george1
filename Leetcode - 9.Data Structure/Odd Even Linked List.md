
**样例1:**
```
输入： 1->2->3->4->5->NULL
输出： 1->3->5->2->4->NULL
```
**样例2:**

```
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