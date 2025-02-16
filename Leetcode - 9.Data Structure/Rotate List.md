Lintcode 170
给定一个链表，旋转链表，使得每个节点向右移动k个位置，其中k是一个非负数

**样例 1:**
```python
"""
输入：1->2->3->4->5  k = 2
输出：4->5->1->2->3
```
**样例 2:**
```python
"""
输入：3->2->1  k = 1
输出：1->3->2
```


```python
def rrotate_right(self, head, k):
	if head==None:
		return head
	curNode = head
	size = 1
	while curNode!=None:
		size += 1
		curNode = curNode.next
	size -= 1
	k = k%size
	if k==0:
		return head
	len = 1
	curNode = head
	while len<size-k:
		len += 1
		curNode = curNode.next
	newHead = curNode.next
	curNode.next = None
	curNode = newHead
	while curNode.next!=None:
		curNode = curNode.next
	curNode.next = head
	return newHead
```
pass


## **LintCode 170: Rotate List**

### **解法分析**

本題要求將鏈表向 **右旋轉 `k` 次**，即每次將鏈表最後一個節點移動到開頭。

**示例**
```python
輸入：
head = 1 → 2 → 3 → 4 → 5
k = 2

步驟：
1. 旋轉 1 次：5 → 1 → 2 → 3 → 4
2. 旋轉 2 次：4 → 5 → 1 → 2 → 3

輸出：
4 → 5 → 1 → 2 → 3

```

---

### **解法思路**

1. **計算鏈表長度 `size`**
    
    - 遍歷鏈表，計算節點數量 `size`。
2. **處理 `k` 大於 `size` 的情況**
    
    - 由於旋轉 `size` 次後，鏈表回到原始狀態，因此 `k = k % size`。
3. **找到新的頭節點**
    
    - 設 `curNode = head`，從頭開始遍歷，走 `size-k-1` 步，找到新的頭 `newHead`。
4. **重新連接鏈表**
    
    - 斷開 `curNode.next`，並將 `newHead` 之前的部分作為尾部。
    - 找到 `newHead` 的最後一個節點，將其 `next` 指向 `head`，完成旋轉。

---

### **變數表**

|變數名稱|角色|作用|初始值|變化過程|
|---|---|---|---|---|
|`head`|鏈表頭部|指向原始鏈表的第一個節點|`head`|旋轉後變成 `newHead`|
|`curNode`|遍歷指標|用來遍歷鏈表|`head`|依據不同步驟更新|
|`size`|鏈表長度|記錄鏈表的總節點數|`1`|遍歷計算|
|`k`|旋轉次數|優化後的旋轉步數|`k`|`k % size`|
|`len`|走過的步數|幫助找到新頭節點|`1`|`len += 1`|
|`newHead`|旋轉後的新頭|旋轉後的第一個節點|`curNode.next`|最後返回|

---

### **時間與空間複雜度分析**

#### **時間複雜度：O(n)**

1. 遍歷鏈表一次 **計算長度**（O(n)）。
2. 遍歷一次 **找到新頭**（O(n)）。
3. 遍歷一次 **找到最後節點**（O(n)）。
4. 總計 **O(n) + O(n) + O(n) = O(n)**。

#### **空間複雜度：O(1)**

- 只使用了 **常數額外變數**，因此 **空間複雜度為 O(1)**。

---

### **其他解法**

1. **閉環+斷開法（O(n) 時間, O(1) 空間）**
    
    - 先將鏈表**變成環**，然後斷開合適的位置，效率更高。
2. **雙指針法（O(n) 時間, O(1) 空間）**
    
    - 使用 **快慢指針** 找到旋轉點，然後重新連接鏈表。
3. **使用棧（O(n) 時間, O(n) 空間）**
    
    - 先把鏈表存進 **棧**，再按照 `k` 次數取出並重新連接。

---

### **結論**

- **最佳解法** 為 **閉環+斷開法**，因為它能夠 **一次遍歷完成旋轉**。
- **本解法雖然有效，但遍歷次數較多**（三次 O(n)），可以進一步優化。