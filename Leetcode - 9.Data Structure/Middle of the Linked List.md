
**样例 1:**
```
输入：1->2->3->4->5->null
输出：3->4->5->null
```
**样例 2:**
```
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