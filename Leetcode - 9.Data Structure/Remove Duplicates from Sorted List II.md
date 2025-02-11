Lintcode 113
给定一个排序链表，删除所有重复的元素，只留下原链表中没有重复的元素。

**样例 1：**
输入：
```python
"""
linked list = null
```
输出：
```python
"""
null
```
解释：
空链表返回null

**样例 2：**
输入：
```python
"""
linked list = 1->2->3->3->4->4->5->null
```
输出：
```python
"""
1->2->5->null
```
解释：
删除所有3和4

**样例 3：**
输入：
```python
"""
linked list = 1->1->1->2->3->null
```
输出：
```python
"""
2->3->null
```
解释：
删除所有1


```python
def delete_duplicates(self, head):  
	if None == head or None == head.next:  
		return head  

	new_head = ListNode(-1)  
	new_head.next = head  
	parent = new_head  
	cur = head  
	while None != cur and None != cur.next:   ### check cur.next None  
		if cur.val == cur.next.val:  
			val = cur.val  
			while None != cur and val == cur.val: ### check cur None  
				cur = cur.next  
			parent.next = cur  
		else:  
			cur = cur.next  
			parent = parent.next  

	return new_head.next 
```
pass