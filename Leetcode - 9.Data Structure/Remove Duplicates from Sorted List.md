Lintcode 112
给定一个排序链表，删除所有重复的元素，每个元素只留下一个。

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
linked list = 1->1->2->null
```
输出：
```python
"""
1->2->null
```
解释：
删除重复的1

**样例 3：**
输入：
```python
"""
linked list = 1->1->2->3->3->null
```
输出：
```python
"""
1->2->3->null
```
解释：
删除重复的1和3


```python
def delete_duplicates(self, head):
	p = head
	while(p != None and p.next != None):
		if p.val != p.next.val:
			p = p.next
		else:
			p.next = p.next.next
	return head
```
pass