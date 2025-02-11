Lintcode 170
给定一个链表，旋转链表，使得每个节点向右移动k个位置，其中k是一个非负数

**样例 1:**
```python
"""
输入：1->2->3->4->5  k = 2
输出：4->5->1->2->3
```
**样例 2:**
```python
"""
输入：3->2->1  k = 1
输出：1->3->2
```


```python
def rrotate_right(self, head, k):
	if head==None:
		return head
	curNode = head
	size = 1
	while curNode!=None:
		size += 1
		curNode = curNode.next
	size -= 1
	k = k%size
	if k==0:
		return head
	len = 1
	curNode = head
	while len<size-k:
		len += 1
		curNode = curNode.next
	newHead = curNode.next
	curNode.next = None
	curNode = newHead
	while curNode.next!=None:
		curNode = curNode.next
	curNode.next = head
	return newHead
```
pass