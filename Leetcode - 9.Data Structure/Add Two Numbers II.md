Lintcode 221
假定用链表表示两个数，其中每个节点仅包含一个数字。假设这两个数的数字`顺序`排列，请设计一种方法将两个数相加，并将其结果表现为链表的形式。

**样例 1:**
```python
"""
输入: 6->1->7   2->9->5
输出: 9->1->2
```
**样例 2:**
```python
"""
输入: 1->2->3   4->5->6
输出: 5->7->9
```


```python
class Solution:
    """
    @param l1: The first list.
    @param l2: The second list.
    @return: the sum list of l1 and l2.
    """
    # 反转链表
    def reverse(self, l):
        # pre->cur反转为cur->pre,next用于遍历原链表
        pre = None
        cur = l
        next = cur.next
        while next:
            cur.next = pre
            pre = cur
            cur = next
            next = next.next
        cur.next = pre
        return cur
      
    def add_lists2(self, l1, l2):
        l1 = self.reverse(l1)
        l2 = self.reverse(l2)
        ans = ListNode(0)
        cur = ans
        # pre用于最后删去最高位为0的结点
        pre = None
        # l1和l2逐位从低位到高位相加，直到l1或l2到最高位
        while l1 and l2:
            # sum = 进位 + 二者之和
            sum = cur.val + l1.val + l2.val
            cur.val = sum % 10
            cur.next = ListNode(sum // 10)
            l1 = l1.next
            l2 = l2.next
            pre = cur
            cur = cur.next
        # 如果l1 或 l2还有更高位，继续加到答案链表
        while l1:
            sum = cur.val + l1.val
            cur.val = sum % 10
            cur.next = ListNode(sum // 10)
            l1 = l1.next
            pre = cur
            cur = cur.next
        while l2:
            sum = cur.val + l2.val
            cur.val = sum % 10;
            cur.next = ListNode(sum // 10)
            l2 = l2.next
            pre = cur
            cur = cur.next
        if cur.val == 0:
            pre.next = cur.next
        return self.reverse(ans)
```
pass

## **LintCode 221: Add Two Numbers II**

### **解法分析**

本題要求對應加總 **兩個單向鏈表所表示的數字**，其中：

- **每個節點存儲一位數**（0-9）。
- **數字是正序存儲**（最高位在鏈表頭部）。
- **輸出仍需以正序存儲**。

---

### **解法思路**

1. **反轉兩個鏈表**
    
    - 由於數字是 **正序存儲**，為了方便從 **低位到高位相加**，我們 **先反轉兩個鏈表**。
    - 例如：
```python
l1: 6 → 1 → 7   （表示 617）
l2: 2 → 9 → 5   （表示 295）
反轉後：
l1: 7 → 1 → 6
l2: 5 → 9 → 2

```
        
2. **從最低位開始相加**
    
    - 用 `cur.val + l1.val + l2.val` 計算 **當前位數**，並將 **進位存入 `cur.next`**。
    - 例如：
```python
7 + 5 = 12，記 2，進位 1
1 + 9 + 1（進位）= 11，記 1，進位 1
6 + 2 + 1（進位）= 9，記 9
```
        
3. **處理剩餘數字**
    
    - 如果 `l1` 或 `l2` 還有剩餘數字，繼續累加進位。
    - 若最高位的進位為 `0`，則刪除該節點。
4. **再次反轉回來**
    
    - 由於結果仍需 **正序存儲**，因此將 **結果鏈表再反轉回去**。

---

### **變數表**

|變數名稱|角色|作用|初始值|變化過程|
|---|---|---|---|---|
|`l1`|第一個數|代表第一個鏈表|`l1`|先反轉後加總|
|`l2`|第二個數|代表第二個鏈表|`l2`|先反轉後加總|
|`pre`|前驅指標|用於刪除最高位 `0`|`None`|指向 `cur`|
|`cur`|遍歷指標|當前加總的節點|`ans`|逐步遍歷|
|`sum`|當前總和|記錄當前位的總和|`cur.val + l1.val + l2.val`|進位後更新|
|`next`|反轉指標|幫助反轉鏈表|`l.next`|逐步遍歷|

---

### **時間與空間複雜度分析**

#### **時間複雜度：O(n)**

1. **反轉兩個鏈表**（O(n)）。
2. **遍歷加總兩個鏈表**（O(n)）。
3. **再次反轉結果鏈表**（O(n)）。
4. **總計：O(n) + O(n) + O(n) = O(n)**。

#### **空間複雜度：O(1)**

- 只使用了 **常數額外變數** (`cur`, `pre`, `sum`)，無額外數據結構，因此 **O(1)**。

---

### **其他解法**

1. **使用棧（Stack, O(n) 時間, O(n) 空間）**
    
    - 先將 `l1` 和 `l2` 存入 **棧**，然後 **從棧頂開始相加**，並 **建立新鏈表** 存儲結果。
2. **遞歸方法（O(n) 時間, O(n) 空間）**
    
    - 用 **遞歸遍歷至尾部**，逐層返回時 **從低位開始相加**。
3. **數字轉換法（O(n) 時間, O(1) 空間）**
    
    - 先將 `l1` 和 `l2` **轉為整數**，加總後拆分回鏈表（不適用大數字）。

---

### **結論**

- **最佳解法為反轉鏈表法**，因為它只需 **O(n) 時間, O(1) 空間**。
- 若允許 **O(n) 空間**，可使用 **棧** 或 **遞歸方法**，實作較為直觀。