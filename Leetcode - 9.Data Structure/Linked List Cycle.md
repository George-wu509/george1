

**样例 1：**
输入：
```
linked list = 21->10->4->5
tail connects to node index 1(value 10).
```
输出：
```
true
```
解释：
链表有环。

**样例 2：**
输入：
```
linked list = 21->10->4->5->null
```
输出：
```
false
```
解释：
链表无环。

```python
def hasCycle(self, head):
	if head is None:			
		return False		
	p1 = head		
	p2 = head		
	while True:
		if p1.next is not None:
			p1=p1.next.next
			p2=p2.next
			if p1 is None or p2 is None:
				return False
			elif p1 == p2:
				return True
		else:
			return False
	return False
```
pass