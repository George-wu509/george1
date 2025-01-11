
**样例 1：**
```
输入：
1->2->3->4->null
3
输出：
1->2->4->null
解释： 
删除链表中值为 3 的节点 node，在调用函数后，最终链表变为 1->2->4->null
```
**样例 2：**
```
输入：
1->3->5->null
3
输出：
1->5->null
解释： 
删除链表中值为 3 的节点 node，在调用函数后，最终链表变为 1->5->null
```


```python
def deleteNode(self, node):
	if not node:
		return
	node.val = node.next.val
	node.next = node.next.next
```
pass