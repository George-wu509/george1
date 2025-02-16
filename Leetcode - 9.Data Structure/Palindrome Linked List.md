Lintcode 223
设计一个函数判断一个链表是否为回文链表。

**样例 1:**
```python
"""
输入: 1->2->1
输出: true
```
**样例 2:**
```python
"""
输入: 2->2->1
输出: false
```


```python
def is_palindrome(self, head):
	if head is None:
		return True

	fast = slow = head
	while fast.next and fast.next.next:
		slow = slow.next
		fast = fast.next.next

	p, last = slow.next, None
	while p:
		next = p.next
		p.next = last
		last, p = p, next

	p1, p2 = last, head
	while p1 and p1.val == p2.val:
		p1, p2 = p1.next, p2.next

	p, last = last, None
	while p:
		next = p.next
		p.next = last
		last, p = p, next
		slow.next = last
	return p1 is None
```
pass


## **LintCode 223: Palindrome Linked List**

### **解法分析**

本題要求判斷一個**單向鏈表**是否為「回文鏈表（Palindrome Linked List）」，即：

- 頭尾對應的元素相等，例如：
    
    複製編輯
    
    `1 → 2 → 3 → 2 → 1  （是回文） 1 → 2 → 3 → 4 → 1  （不是回文）`
    

---

### **解法思路**

1. **利用快慢指針找出中點**
    
    - `fast` 每次移動 2 步，`slow` 每次移動 1 步。
    - 當 `fast` 到達尾部時，`slow` 會停在**中間節點**。
2. **反轉後半部鏈表**
    
    - 使用 `last`（上一步的節點） 和 `p`（當前節點） 進行反轉，讓後半部倒序。
3. **比較前半部與反轉後的後半部**
    
    - 用 `p1` 遍歷**反轉後的後半部**，`p2` 遍歷**前半部**，逐個比較值是否相等。
4. **恢復原始鏈表**
    
    - 重新反轉後半部鏈表，恢復原來的鏈表結構。

---

### **變數表**

|變數名稱|角色|作用|初始值|變化過程|
|---|---|---|---|---|
|`head`|鏈表頭|指向整個鏈表的起始節點|`head`|不變|
|`fast`|快指針|用於尋找中點，每次前進 2 步|`head`|遍歷至尾部|
|`slow`|慢指針|用於尋找中點，每次前進 1 步|`head`|遍歷至中點|
|`p`|當前反轉指針|用於反轉後半部|`slow.next`|遍歷至 `None`|
|`last`|反轉後的頭節點|幫助反轉後半部|`None`|最終指向反轉後的頭|
|`p1`|指向反轉後的頭|用於比較後半部鏈表|`last`|遍歷至 `None`|
|`p2`|指向前半部|用於比較前半部鏈表|`head`|遍歷至 `None`|

---

### **時間與空間複雜度分析**

#### **時間複雜度：O(n)**

1. 找中點（O(n)）
2. 反轉後半部（O(n)）
3. 比較兩半（O(n)）
4. 還原後半部（O(n)）

總計為 **O(n)**。

#### **空間複雜度：O(1)**

- 只用了 **幾個額外變數** (`fast`, `slow`, `p`, `last`)，無額外空間需求，因此 **O(1)**。

---

### **其他解法**

1. **使用棧（Stack, O(n) 時間, O(n) 空間）**
    
    - 遍歷鏈表，將值壓入**棧**，再逐個比對是否相等。
2. **使用遞歸（O(n) 時間, O(n) 空間）**
    
    - 用遞歸方式檢查頭尾值是否相等。
3. **雙指針 + 雙端隊列（O(n) 時間, O(n) 空間）**
    
    - 使用 `collections.deque` 將鏈表轉成雙端隊列，從兩端比較。

---

### **結論**

- **最佳解法為雙指針反轉法**，因為它只需 **O(n) 時間, O(1) 空間**，不使用額外數據結構，且可恢復原始鏈表結構。
- **若允許 O(n) 空間**，則可使用 **棧** 或 **雙端隊列** 方法，實作較為直觀。