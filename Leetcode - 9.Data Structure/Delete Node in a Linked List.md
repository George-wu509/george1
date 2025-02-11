Lintcode 372
给定一个单链表 `head` 和一个等待被删除的节点 `node`（非表头或表尾）。
你 **无法访问** 第一个节点 `head`，请在 O(1) 的时间复杂度删除该链表节点 `node`。

**样例 1：**
```python
"""
输入：
1->2->3->4->null
3
输出：
1->2->4->null
解释： 
删除链表中值为 3 的节点 node，在调用函数后，最终链表变为 1->2->4->null
```
**样例 2：**
```python
"""
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