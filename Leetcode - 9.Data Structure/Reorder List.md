Lintcode 99
给定一个单链表L: _L_0→_L_1→…→_L_n-1→_L_n,

重新排列后为：_L_0→_L_n→_L_1→_L_n-1→_L_2→_L_n-2→…

必须在不改变节点值的情况下进行原地操作。

**样例 1：**
输入：
```python
"""
list = 1->2->3->4->null
```
输出：
```python
"""
1->4->2->3->null
```
解释：
按照规则重新排列即可 

**样例 2：**
输入：
```python
"""
list = 1->2->3->4->5->null
```
输出：
```python
"""
1->5->2->4->3->null
```
解释：
按照规则重新排列即可


```python
def reorder_list(self, head):
	if None == head or None == head.next:
		return head

	pfast = head
	pslow = head
	
	while pfast.next and pfast.next.next:
		pfast = pfast.next.next
		pslow = pslow.next
	pfast = pslow.next
	pslow.next = None
	
	pnext = pfast.next
	pfast.next = None
	while pnext:
		q = pnext.next
		pnext.next = pfast
		pfast = pnext
		pnext = q

	tail = head
	while pfast:
		pnext = pfast.next
		pfast.next = tail.next
		tail.next = pfast
		tail = tail.next.next
		pfast = pnext
	return head
```
pass