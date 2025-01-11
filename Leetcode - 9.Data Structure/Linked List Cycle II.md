
**样例 1：**
输入：
```
null，no cycle
```
输出：
```
no cycle
```
解释：
链表为空，所以没有环存在。

**样例 2：**
输入：
```
-21->10->4->5，tail connects to node index 1
```
输出：
```
10
```
解释：
最后一个节点5指向下标为1的节点，也就是10，所以环的入口为10。


```python
def detectCycle(self, head):
	if head == None or head.next == None:
		return None
	slow = fast = head  		#初始化快指针和慢指针
	while fast and fast.next:	
		slow = slow.next
		fast = fast.next.next
		if fast == slow:		#快慢指针相遇
			break
	if slow == fast:
		slow = head				#从头移动慢指针
		while slow != fast:
			slow = slow.next
			fast = fast.next
		return slow				#两指针相遇处即为环的入口
	return None
```
pass