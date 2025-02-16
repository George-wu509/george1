Lintcode 104
合并 `k` 个排序链表（序列为升序序列），并且返回合并后的排序链表（序列为升序序列）。尝试分析和描述其复杂度。

**样例 1：**
输入：
```python
"""
lists = [2->4->null,null,-1->null]
```
输出：
```python
"""
-1->2->4->null
```
解释：
将2->4->null、null和-1->null合并成一个升序的链表。

**样例 2：**
输入：
```python
"""
lists = [2->6->null,5->null,7->null]
```
输出：
```python
"""
2->5->6->7->null
```
解释：
将2->6->null、5->null和7->null合并成一个升序的链表。

```python
class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        if not lists:
            return None
        
        return self.merge_range_lists(lists, 0, len(lists) - 1)
        
    def merge_range_lists(self, lists, start, end):
        if start == end:
            return lists[start]
        
        mid = (start + end) // 2
        left = self.merge_range_lists(lists, start, mid)
        right = self.merge_range_lists(lists, mid + 1, end)
        return self.merge_two_lists(left, right)
        
    def merge_two_lists(self, head1, head2):
        tail = dummy = ListNode(0)
        while head1 and head2:
            if head1.val < head2.val:
                tail.next = head1
                head1 = head1.next
            else:
                tail.next = head2
                head2 = head2.next
            tail = tail.next
            
        if head1:
            tail.next = head1
        if head2:
            tail.next = head2
        
        return dummy.next
```
pass

## **LintCode 104: Merge K Sorted Lists**

### **解法分析**

本題要求合併 **K 個排序好的鏈表**，並返回一個 **排序後的單一鏈表**。

#### **範例**
```python
輸入：
lists = [
  1 → 4 → 5,
  1 → 3 → 4,
  2 → 6
]

輸出：
1 → 1 → 2 → 3 → 4 → 4 → 5 → 6

```
---

### **解法思路**

本解法採用 **分治法（Divide and Conquer）**，核心思路是：

1. **遞歸拆分問題**
    
    - 將 `lists` 劃分為左右兩部分，分別遞歸合併。
    - **直到 `start == end`，則返回該單一鏈表**。
2. **合併兩個排序鏈表**
    
    - 依照 **歸併排序（Merge Sort）** 的方法，使用 **雙指針遍歷 `head1` 和 `head2`**，將較小的節點連接到 `dummy` 節點後。
3. **最終返回合併後的鏈表**
    

---

### **變數表**

|變數名稱|角色|作用|初始值|變化過程|
|---|---|---|---|---|
|`lists`|輸入鏈表列表|包含 `K` 個有序鏈表|`lists`|被分割成左右兩部分|
|`start`|左邊界|標記遞歸的開始索引|`0`|遞歸變化|
|`end`|右邊界|標記遞歸的結束索引|`len(lists) - 1`|遞歸變化|
|`mid`|中點|拆分 `lists` 為左右兩部分|`(start + end) // 2`|遞歸計算|
|`left`|左半部分合併後的鏈表|存放左側合併結果|`lists[start]`|遞歸合併|
|`right`|右半部分合併後的鏈表|存放右側合併結果|`lists[mid + 1]`|遞歸合併|
|`dummy`|虛擬節點|幫助合併 `head1` 和 `head2`|`ListNode(0)`|連接排序節點|
|`tail`|合併後的當前節點|指向 `dummy.next` 的最後一個節點|`dummy`|遍歷 `head1` 和 `head2`|

---

### **時間與空間複雜度分析**

#### **時間複雜度：O(N log K)**

- **每次合併的時間為 O(N)**（`N` 是總節點數量）。
- **遞歸深度為 O(log K)**（每次遞歸將 `K` 個鏈表分成兩半）。
- **總計 O(N log K)**。

#### **空間複雜度：O(log K)**

- 由於使用 **遞歸**，遞歸深度為 **O(log K)**，所以空間複雜度為 **O(log K)**。

---

### **其他解法**

4. **使用最小堆（Priority Queue, O(N log K) 時間, O(K) 空間）**
    
    - 使用 **最小堆（Min-Heap）** 存入 `K` 個鏈表的頭節點，每次彈出最小節點並插入新節點。
5. **逐個合併（O(NK) 時間, O(1) 空間）**
    
    - 先合併 `lists[0]` 和 `lists[1]`，再將 `lists[2]` 併入，依此類推（效率較低）。

---

### **結論**

- **最佳解法為分治法（Divide and Conquer）**，因為它 **O(N log K) 時間, O(log K) 空間**。
- 若允許 **O(K) 空間**，則可使用 **最小堆（Priority Queue）**，但 `log K` 深度遞歸仍然較優。