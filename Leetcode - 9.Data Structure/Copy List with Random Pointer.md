
Lintcode 105
给出一个链表，每个节点包含一个额外增加的随机指针，其可以指向链表中的任何节点或空的节点。  
返回其链表的深度拷贝。


```python
class Solution:
    # @param head: A RandomListNode
    # @return: A RandomListNode
    def copyRandomList(self, head):
        # write your code here
        if head == None:
            return None
            
        myMap = {}
        nHead = RandomListNode(head.label)
        myMap[head] = nHead
        p = head
        q = nHead
        while p != None:
            q.random = p.random
            if p.next != None:
                q.next = RandomListNode(p.next.label)
                myMap[p.next] = q.next
            else:
                q.next = None
            p = p.next
            q = q.next
        
        p = nHead
        while p!= None:
            if p.random != None:
                p.random = myMap[p.random]
            p = p.next
        return nHead
```
pass


## **LintCode 105: Copy List with Random Pointer**

### **解法分析**

本題要求 **複製帶有隨機指針（random pointer）的單向鏈表**，其中：

- 每個節點有兩個指針：
    - `next` 指向 **下一個節點**。
    - `random` 指向 **鏈表內任意節點（或 `None`）**。

#### **範例**
```python
原始鏈表：
A → B → C → D
|    |    |    |
v    v    v    v
C    A    None B

複製後：
A' → B' → C' → D'
|     |     |     |
v     v     v     v
C'    A'    None  B'

```

---

### **解法思路**

本方法 **使用哈希表（Dictionary）記錄原始節點與新節點的映射**，分兩步進行：

1. **第一輪遍歷**：複製 `next` 指針，並建立 `myMap`（字典映射原節點與新節點）。
2. **第二輪遍歷**：設定 `random` 指針。

---

### **解法步驟**

1. **處理邊界條件**
    
    - 若 `head == None`，直接返回 `None`。
2. **建立哈希表**
    
    - 建立 `myMap = {}` 存儲原鏈表節點 → 新鏈表節點的映射。
3. **第一輪遍歷：複製 `next` 指針**
    
    - 從 `head` 開始遍歷：
        - **創建新節點** `q = RandomListNode(p.label)` 並存入 `myMap`。
        - **連接 `next` 指針**（讓 `q.next = 新節點`）。
        - 移動 `p` 和 `q` 指針至下一個節點。
4. **第二輪遍歷：複製 `random` 指針**
    
    - 遍歷新鏈表 `p = nHead`，查找 `myMap[p.random]` 並設置 `random` 指針。
5. **返回新鏈表頭 `nHead`**。
    

---

### **變數表**

|變數名稱|角色|作用|初始值|變化過程|
|---|---|---|---|---|
|`head`|原始鏈表頭|指向原始鏈表|`head`|不變|
|`myMap`|哈希表|存儲 **原節點 → 新節點** 的映射|`{}`|逐步增加|
|`p`|遍歷指標|遍歷原鏈表|`head`|依次移動到 `p.next`|
|`q`|遍歷指標|遍歷新鏈表|`nHead`|依次移動到 `q.next`|
|`nHead`|新鏈表頭|指向複製後的鏈表|`RandomListNode(head.label)`|不變|

---

### **時間與空間複雜度分析**

#### **時間複雜度：O(n)**

- **第一輪遍歷 O(n)**：複製 `next` 指針並建立映射。
- **第二輪遍歷 O(n)**：複製 `random` 指針。
- **總計 O(n) + O(n) = O(n)**。

#### **空間複雜度：O(n)**

- **哈希表 `myMap`** 需要存儲 `n` 個節點的映射，因此 **空間複雜度為 O(n)**。

---

### **其他解法**

6. **O(1) 空間的複製法（雙指針+交錯鏈表）**
    
    - **第一輪遍歷**：將複製節點插入原始鏈表中。
    - **第二輪遍歷**：設置 `random` 指針。
    - **第三輪遍歷**：拆分原鏈表與複製鏈表。
7. **遞歸解法（O(n) 時間, O(n) 空間）**
    
    - 使用遞歸方式複製 `next`，然後遞歸設置 `random`。

---

### **結論**

- **最優解為 O(1) 空間複製法**，因為它只需 **O(n) 時間, O(1) 空間**。
- **哈希表方法（本解法）簡單易懂，但需要 O(n) 額外空間**。