
**样例 1：**
输入：
```
linked list = null
```
输出：
```
null
```
解释：
空链表返回null

**样例 2：**
输入：
```
linked list = 1->1->2->null
```
输出：
```
1->2->null
```
解释：
删除重复的1

**样例 3：**
输入：
```
linked list = 1->1->2->3->3->null
```
输出：
```
1->2->3->null
```
解释：
删除重复的1和3


```python
def delete_duplicates(self, head):
	flag = 1
	p = head
	while(p != None and p.next != None):
		if p.val != p.next.val:
			flag = 1
			p = p.next
		else:
			p.next = p.next.next
	return head
```
pass