Lintcode 35
翻转一个链表


**样例 1：**
输入：
```python
"""
链表 = 1->2->3->null
```
输出：
```python
"""
3->2->1->null
```
解释：
翻转链表

**样例 2：**
输入：
```python
"""
链表 = 1->2->3->4->null
```
输出：
```python
"""
4->3->2->1->null
```
解释：
翻转链表


```python
    def reverse(self, head):
        #curt表示前继节点
        curt = None
        while head != None:
            #temp记录下一个节点，head是当前节点
            temp = head.next
            head.next = curt
            curt = head
            head = temp
        return curt
```
pass