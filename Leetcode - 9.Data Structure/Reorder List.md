Lintcode 99
给定一个单链表L: _L_0→_L_1→…→_L_n-1→_L_n,

重新排列后为：_L_0→_L_n→_L_1→_L_n-1→_L_2→_L_n-2→…

必须在不改变节点值的情况下进行原地操作。

**样例 1：**
输入：
```python
"""
list = 1->2->3->4->null
```
输出：
```python
"""
1->4->2->3->null
```
解释：
按照规则重新排列即可 

**样例 2：**
输入：
```python
"""
list = 1->2->3->4->5->null
```
输出：
```python
"""
1->5->2->4->3->null
```
解释：
按照规则重新排列即可


```python
def reorder_list(self, head):
	if None == head or None == head.next:
		return head

	pfast = head
	pslow = head
	
	while pfast.next and pfast.next.next:
		pfast = pfast.next.next
		pslow = pslow.next
	pfast = pslow.next
	pslow.next = None
	
	pnext = pfast.next
	pfast.next = None
	while pnext:
		q = pnext.next
		pnext.next = pfast
		pfast = pnext
		pnext = q

	tail = head
	while pfast:
		pnext = pfast.next
		pfast.next = tail.next
		tail.next = pfast
		tail = tail.next.next
		pfast = pnext
	return head
```
pass


## **LintCode 99: Reorder List**

### **解法分析**

本題要求對 **單向鏈表** 進行 **重排**，使得：

- **首尾交錯排列**，原始順序為：

    `L0 → L1 → L2 → ... → Ln`
    
    轉換後變為：

    `L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ...`
    

#### **範例**
```python
輸入：
1 → 2 → 3 → 4 → 5

輸出：
1 → 5 → 2 → 4 → 3

```

---

### **解法思路**

本解法 **使用快慢指針 + 反轉後半部分 + 交錯合併** 來完成重排：

1. **使用快慢指針找到中間節點**
    
    - `pfast` 每次前進 **2 步**，`pslow` 每次前進 **1 步**。
    - `pslow` 最終會停在 **鏈表的中間節點**。
2. **反轉後半部分鏈表**
    
    - 以 `pslow.next` 為新的頭部，使用 **迭代法** 進行反轉。
3. **交錯合併**
    
    - 逐個交錯合併 **前半部分與反轉後的後半部分**，形成新順序。

---

### **變數表**

|變數名稱|角色|作用|初始值|變化過程|
|---|---|---|---|---|
|`head`|鏈表頭|指向原始鏈表的第一個節點|`head`|不變|
|`pfast`|快指針|每次前進 2 步，找到中點|`head`|遍歷至 `None` 或 `pslow.next`|
|`pslow`|慢指針|每次前進 1 步，找到中點|`head`|遍歷至中間節點|
|`pnext`|反轉指標|幫助反轉後半部分|`pfast.next`|遍歷至 `None`|
|`q`|暫存指標|用於記錄下一個節點|`pnext.next`|遍歷至 `None`|
|`tail`|前半部分指標|幫助合併鏈表|`head`|遍歷至 `None`|

---

### **時間與空間複雜度分析**

#### **時間複雜度：O(n)**

- **第一步：尋找中點（O(n)）**
- **第二步：反轉後半部分（O(n)）**
- **第三步：合併鏈表（O(n)）**
- **總計 O(n) + O(n) + O(n) = O(n)**。

#### **空間複雜度：O(1)**

- 只使用了幾個額外變數，因此 **空間複雜度為 O(1)**。

---

### **其他解法**

4. **使用棧（O(n) 時間, O(n) 空間）**
    
    - 先將鏈表節點壓入 **棧**，再按交錯順序彈出並重建鏈表。
5. **使用數組（O(n) 時間, O(n) 空間）**
    
    - 先將鏈表存入 **數組**，然後利用 **雙指針法** 交錯合併。

---

### **結論**

- **最佳解法為快慢指針 + 反轉後半部 + 交錯合併**，因為它 **O(n) 時間, O(1) 空間**。
- 若允許 **O(n) 空間**，可以使用 **棧** 或 **數組** 方法，但這不是最優解。