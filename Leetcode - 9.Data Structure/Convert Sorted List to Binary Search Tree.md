Lintcode 106
给出一个所有元素以升序排序的单链表，将它转换成一棵高度平衡的二叉搜索树

**样例 1：**
输入：
```python
"""
linked list = 1->2->3
```
输出：
```python
"""
  2  
 / \
1   3
```
解释：
将链表转换成一棵高度平衡的二叉搜索树。

**样例 2：**
输入：
```python
"""
linked list = 2->3->6->7
```
输出：
```python
"""
   3
  / \
 2   6
      \
       7
```
解释：
可能有多个答案，您可以返回任何一个。


```python
class Solution:
    """
    @param: head: The first node of linked list.
    @return: a tree node
    """
    def sorted_list_to_b_s_t(self, head):
        length = self.get_linked_list_length(head)
        root, next = self.convert(head, length)
        return root
    
    def get_linked_list_length(self, head):
        length = 0
        while head:
            length += 1
            head = head.next
        return length
        
    def convert(self, head, length):
        if length == 0:
            return None, head
        
        left_root, middle = self.convert(head, length // 2)
        right_root, next = self.convert(middle.next, length - length // 2 - 1)
        
        root = TreeNode(middle.val)
        root.left = left_root
        root.right = right_root
        
        return root, next
```
pass


## **LintCode 106: Convert Sorted List to Binary Search Tree**

### **解法分析**

本題要求將 **有序單向鏈表** 轉換為 **平衡二元搜尋樹（BST）**，其中：

- **鏈表已經排序**（單調遞增）。
- **BST 需要平衡**（即左右子樹的高度差不超過 `1`）。

---

### **解法思路**

1. **計算鏈表長度**
    
    - 先遍歷鏈表計算長度 `length`，這樣我們知道 **有多少個節點可以構建 BST**。
2. **遞歸構建 BST**
    
    - 使用 **中序遍歷（in-order traversal）方式建立 BST**：
        - **左子樹遞歸構建**：處理前 `length // 2` 個節點。
        - **根節點**：當前 `head`（中間節點）。
        - **右子樹遞歸構建**：處理剩餘 `(length - length // 2 - 1)` 個節點。
3. **遞歸返回**
    
    - 每次遞歸返回 **當前根節點 `root` 及下一個待處理的 `head`**。

---

### **變數表**

|變數名稱|角色|作用|初始值|變化過程|
|---|---|---|---|---|
|`head`|原始鏈表|指向當前處理的節點|`head`|逐步向後移動|
|`length`|鏈表長度|記錄鏈表的總長度|`0`|遍歷計算|
|`root`|BST 根節點|構建二元搜尋樹|`None`|遞歸設置|
|`left_root`|左子樹根節點|構建左子樹|`None`|遞歸設置|
|`right_root`|右子樹根節點|構建右子樹|`None`|遞歸設置|
|`middle`|當前根節點|記錄當前處理的節點|`head`|遞歸移動|
|`next`|下一個節點|指向鏈表下一個未處理節點|`head.next`|依據 `middle.next` 變化|

---

### **時間與空間複雜度分析**

#### **時間複雜度：O(n)**

- **計算鏈表長度（O(n)）**
- **遞歸建立 BST（O(n)）**
- **總計 O(n) + O(n) = O(n)**

#### **空間複雜度：O(log n)**

- **遞歸深度 O(log n)**（遞歸棧空間）。
- **額外變數 O(1)**。

---

### **其他解法**

4. **快慢指針法（O(n log n) 時間, O(log n) 空間）**
    
    - 使用 **快慢指針** 找到中間節點作為根節點，然後 **遞歸構建左、右子樹**。
5. **陣列轉換法（O(n) 時間, O(n) 空間）**
    
    - 先將鏈表存入 **數組（Array）**，然後用 **二分法** 建立 BST。

---

### **結論**

- **最佳解法為 O(n) 時間遞歸法**，因為它**避免了額外的空間開銷**，只使用 **O(log n) 遞歸棧空間**。
- **快慢指針法** 更直觀，但時間複雜度稍高 **O(n log n)**。