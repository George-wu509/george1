Lintcode 102
给定一个链表，判断它是否有环。

**样例 1：**
输入：
```python
"""
linked list = 21->10->4->5
tail connects to node index 1(value 10).
```
输出：
```python
"""
true
```
解释：
链表有环。

**样例 2：**
输入：
```python
"""
linked list = 21->10->4->5->null
```
输出：
```python
"""
false
```
解释：
链表无环。

```python
def hasCycle(self, head):
	if head is None:			
		return False		
	p1 = head		
	p2 = head		
	while True:
		if p1.next is not None:
			p1=p1.next.next
			p2=p2.next
			if p1 is None or p2 is None:
				return False
			elif p1 == p2:
				return True
		else:
			return False
	return False
```
pass


## **LintCode 102: Linked List Cycle**

### **解法分析**

本題要求判斷 **單向鏈表** 是否存在 **環（cycle）**。  
即是否存在某個節點，使得鏈表中的某個節點的 `next` 指向該節點，導致無限循環。

#### **範例**
```python
輸入：
1 → 2 → 3 → 4 → 5 → 2 (回到2)

輸出：
True（存在環）

輸入：
1 → 2 → 3 → 4 → 5 → None

輸出：
False（無環）

```

---

### **解法思路**

**使用快慢指針（Floyd’s Cycle Detection Algorithm, Tortoise and Hare Algorithm）**

1. **初始化**
    
    - `p1`（快指針）：每次移動 **2 步**。
    - `p2`（慢指針）：每次移動 **1 步**。
2. **遍歷鏈表**
    
    - 若 `p1` **先到達 `None`，則鏈表無環**，返回 `False`。
    - 若 `p1 == p2`，則說明 **快指針追上慢指針，鏈表存在環**，返回 `True`。

---

### **變數表**

|變數名稱|角色|作用|初始值|變化過程|
|---|---|---|---|---|
|`head`|鏈表頭|指向鏈表的第一個節點|`head`|不變|
|`p1`|快指針|每次前進 2 步|`head`|遍歷至 `None` 或與 `p2` 相遇|
|`p2`|慢指針|每次前進 1 步|`head`|遍歷至 `p1` 相遇|

---

### **時間與空間複雜度分析**

#### **時間複雜度：O(n)**

- 在最壞情況下，`p1` 和 `p2` 需要遍歷整個鏈表一次，因此時間複雜度為 **O(n)**。
- 若鏈表有環，則 `p1` 在 **O(n)** 內追上 `p2`。

#### **空間複雜度：O(1)**

- 只使用了兩個變數 `p1` 和 `p2`，因此空間複雜度為 **O(1)**。

---

### **其他解法**

1. **使用哈希表（Set, O(n) 時間, O(n) 空間）**
    
    - 遍歷鏈表時，將節點存入 `set`，若某個節點已經出現過，則說明有環。
2. **修改 `next` 指針（O(n) 時間, O(1) 空間，破壞原結構）**
    
    - 遍歷鏈表時將 `next` 指針改為 `None`，若發現 `next` 已為 `None`，則說明有環。

---

### **結論**

- **最優解為快慢指針法**，因為它只需 **O(n) 時間, O(1) 空間**。
- 若允許 **O(n) 空間**，可以使用 **哈希表記錄訪問節點**，但這不是最優解。