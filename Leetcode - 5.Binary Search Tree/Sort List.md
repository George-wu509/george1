
**样例 1：**
输入：
```
list = 1->3->2->null
```
输出：
```
1->2->3->null
```
解释：
给链表排序.

**样例 2：**
输入：
```
list = 1->7->2->6->null
```
输出：
```
1->2->6->7->null
```
解释：
给链表排序.



```python
class Solution:
    """
    @param head: The first node of the linked list.
    @return: You should return the head of the sorted linked list,
                  using constant space complexity.
    """
    def sort_list(self, head):
        # write your code here
        def merge(list1,list2):
            if list1 == None:
                return list2
            if list2 == None:
                return list1
        
            head = None
            
            if list1.val < list2.val:
                head = list1
                list1 = list1.next
            else:
                head = list2;
                list2 = list2.next;
            
            tmp = head
        
            while list1 != None and list2 != None:
                if list1.val < list2.val:
                    tmp.next = list1
                    tmp = list1
                    list1 = list1.next
                else:
                    tmp.next = list2
                    tmp = list2
                    list2 = list2.next
            
            if list1 != None :
                tmp.next = list1;
            if list2 != None :
                tmp.next = list2;
            
            return head;
            
        if head == None:
            return head
        if head.next == None:
            return head
            
        fast = head
        slow = head
        
        while fast.next != None and fast.next.next != None:
            fast = fast.next.next
            slow = slow.next
        
        mid = slow.next
        slow.next = None
        
        list1 = self.sort_list(head)
        list2 = self.sort_list(mid)
        
        sorted = merge(list1,list2)
        
        return sorted
```
pass