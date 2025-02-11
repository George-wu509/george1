Lintcode 1609
给定一个带有头结点 head 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。

**样例 1:**
```python
"""
输入：1->2->3->4->5->null
输出：3->4->5->null
```
**样例 2:**
```python
"""
输入：1->2->3->4->5->6->null
输出：4->5->6->null
```


```python
def middle_node(self, head: ListNode) -> ListNode:
	A = [head]
	while A[-1].next:
		A.append(A[-1].next)
	return A[len(A) // 2]
```
pass