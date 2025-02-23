Lintcode 960
我们需要实现一个叫 `DataStream` 的数据结构。并且这里有 `两` 个方法需要实现：

1. `void add(number)` // 加一个新的数
2. `int firstUnique()` // 返回第一个独特的数
例1:

```python
输入:
add(1)
add(2)
firstUnique()
add(1)
firstUnique()
输出:
[1,2]
```
例2:
```python
输入:
add(1)
add(2)
add(3)
add(4)
add(5)
firstUnique()
add(1)
firstUnique()
add(2)
firstUnique()
add(3)
firstUnique()
add(4)
firstUnique()
add(5)
add(6)
firstUnique()
输出:
[1,2,3,4,5,6]
```


```python
class DataStream:

    def __init__(self):
        self.dummy = ListNode(0)
        self.tail = self.dummy
        self.num_to_prev = {}
        self.duplicates = set()
          
    """
    @param num: next number in stream
    @return: nothing
    """
    def add(self, num):
        if num in self.duplicates:
            return
        
        if num not in self.num_to_prev:
            self.push_back(num)            
            return
        
        # find duplicate, remove it from hash & linked list
        self.duplicates.add(num)
        self.remove(num)
    
    def remove(self, num):
        prev = self.num_to_prev.get(num)
        del self.num_to_prev[num]
        prev.next = prev.next.next
        if prev.next:
            self.num_to_prev[prev.next.val] = prev
        else:
            # if we removed the tail node, prev will be the new tail
            self.tail = prev

    def push_back(self, num):
        # new num add to the tail
        self.tail.next = ListNode(num)
        self.num_to_prev[num] = self.tail
        self.tail = self.tail.next

    """
    @return: the first unique number in stream
    """
    def firstUnique(self):
        if not self.dummy.next:
            return None
        return self.dummy.next.val
```
pass


# **LintCode 960: First Unique Number in Data Stream II 解法分析**

## **解題目標**

實作一個 `DataStream` 類，支援 **動態插入數字**，並能夠 **即時查詢** **當前第一個唯一數字**（即數據流中最早出現且目前仍唯一的數字）。

支援以下操作：

1. `add(num)`: **新增數字 `num` 到數據流中**，若 `num` 出現過則標記為重複。
2. `firstUnique()`: **返回當前數據流中的第一個唯一數字**，若無則返回 `None`。

---

## **解法核心**

這是一個 **動態數據流處理問題**，需要：

1. **維護數據插入順序**。
2. **即時查詢第一個唯一數字**（時間複雜度 `O(1)`）。
3. **即時移除重複數字**（避免線性遍歷）。

### **為何選擇「雙向鏈表 + 哈希表」？**

**關鍵問題**

- **哈希表 (`dict`) 無法保持數字插入順序**
- **普通 `list` 查找唯一數字 `O(n)`，不夠快**
- **`OrderedDict` 雖能保持順序，但刪除操作較慢 (`O(n)`)**

✅ **使用「雙向鏈表 + 哈希表」來解決問題**

- **雙向鏈表** (`LinkedList`) **維護數字插入順序**。
- **哈希表 (`dict`)** 來記錄數字的位置，確保刪除 `O(1)`。
- **額外使用 `set()` 來記錄重複數字**，加速查詢。

---

## **解法步驟**

### **Step 1: 初始化**

python

複製編輯

``def __init__(self):     self.dummy = ListNode(0)  # 虛擬頭節點 (dummy head)     self.tail = self.dummy  # 初始化 tail 指向 dummy     self.num_to_prev = {}  # 記錄數字對應的「前一個節點」，用來 `O(1)` 刪除     self.duplicates = set()  # 記錄已出現過多次的數字``

- **雙向鏈表** 用於維護唯一數字的順序
- **`num_to_prev` (`dict`) 記錄每個數字在鏈表中的「前驅節點」，方便 `O(1)` 刪除
- **`duplicates` (`set`) 記錄所有重複出現的數字，防止誤添加**

---

### **Step 2: `add(num)` 插入數字**

python

複製編輯

`def add(self, num):     if num in self.duplicates:         return  # 直接跳過          if num not in self.num_to_prev:         self.push_back(num)  # 新數字，加入鏈表         return      # 數字出現過 -> 記錄進 duplicates，並從鏈表中刪除     self.duplicates.add(num)     self.remove(num)`

#### **處理三種情況**

1. **`num` 已經是重複數字 (`duplicates` 中)**，直接忽略。
2. **`num` 是第一次出現**，加入鏈表尾部。
3. **`num` 已出現過一次**，標記為重複並從鏈表中刪除。

---

### **Step 3: `remove(num)` 刪除數字**

python

複製編輯

`def remove(self, num):     prev = self.num_to_prev.get(num)     del self.num_to_prev[num]  # 從哈希表移除記錄          prev.next = prev.next.next  # 跳過當前節點     if prev.next:         self.num_to_prev[prev.next.val] = prev  # 更新哈希表     else:         self.tail = prev  # 若刪除的是尾節點，更新 tail`

#### **刪除過程**

- 透過 `num_to_prev[num]` 取得 `num` 的前驅節點 `prev`。
- **直接跳過 `num` 節點**，讓 `prev.next = prev.next.next`，達成 `O(1)` 刪除。
- **如果 `num` 是最後一個節點**，則更新 `tail`。

---

### **Step 4: `push_back(num)` 添加數字到鏈表尾部**

python

複製編輯

`def push_back(self, num):     self.tail.next = ListNode(num)  # 創建新節點     self.num_to_prev[num] = self.tail  # 記錄前驅節點     self.tail = self.tail.next  # 更新 tail`

#### **為何使用 `tail` 插入？**

- **避免 `O(n)` 遍歷**：普通 `list.append()` 在刪除時可能導致 `O(n)`。
- **通過 `tail.next` 直接插入**，時間複雜度 `O(1)`。

---

### **Step 5: `firstUnique()` 查詢第一個唯一數字**

python

複製編輯

`def firstUnique(self):     if not self.dummy.next:         return None  # 無唯一數字     return self.dummy.next.val  # 返回鏈表頭部的數字`

- **唯一數字存於 `dummy.next`**，直接返回即可，時間 `O(1)`。

---

## **變數定義**

|變數名稱|作用|
|---|---|
|`dummy`|虛擬頭節點 (`ListNode(0)`)|
|`tail`|指向當前鏈表的最後一個節點|
|`num_to_prev`|哈希表，記錄 `num` 在鏈表中的 **前驅節點** (`O(1)` 刪除用)|
|`duplicates`|`set()`，存放所有重複數字|
|`num`|當前處理的數字|

---

## **具體範例**

### **範例 1**

text

複製編輯

`操作: add(1), add(2), add(3), firstUnique(), add(2), firstUnique()`

#### **Step 1: 插入 `1, 2, 3`**

text

複製編輯

`dummy → 1 → 2 → 3 firstUnique() = 1`

#### **Step 2: 插入 `2`（標記為重複並刪除）**

text

複製編輯

`dummy → 1 → 3 firstUnique() = 1`

**輸出**

text

複製編輯

`[1, 1]`

---

### **範例 2**

text

複製編輯

`操作: add(7), add(7), firstUnique(), add(3), add(3), firstUnique()`

#### **Step 1: 插入 `7`**

text

複製編輯

`dummy → 7 firstUnique() = 7`

#### **Step 2: 插入 `7`（重複，刪除）**

text

複製編輯

`dummy firstUnique() = None`

#### **Step 3: 插入 `3, 3`**

text

複製編輯

`dummy firstUnique() = None`

**輸出**

text

複製編輯

`[7, None]`

---

## **時間與空間複雜度分析**

### **時間複雜度**

|操作|時間複雜度|說明|
|---|---|---|
|`add(num)`|`O(1)`|插入、刪除均為 `O(1)`|
|`remove(num)`|`O(1)`|利用 `num_to_prev` 快速刪除|
|`push_back(num)`|`O(1)`|直接插入鏈表尾部|
|`firstUnique()`|`O(1)`|直接返回 `dummy.next`|
|**總計**|`O(1)`|所有操作皆為常數時間|

### **空間複雜度**

- `O(n)`，主要來自 `num_to_prev`、`duplicates`、鏈表 `O(n)`。

---

## **其他解法 (不寫 Code)**

1. **`OrderedDict`**
    
    - **優勢**：能夠保持插入順序，刪除時間 `O(1)`。
    - **劣勢**：Python 內建 `OrderedDict` 在某些場景下仍可能 `O(n)`。
2. **普通 `list`**
    
    - **問題**：刪除元素時需要 `O(n)` 遍歷，效率較低。

---

## **總結**

|**解法**|**時間複雜度**|**適用場景**|**優缺點**|
|---|---|---|---|
|**雙向鏈表 + 哈希表 (`O(1)`)**|`O(1)`|最優解|✅ 插入、刪除皆快|
|**`OrderedDict` (`O(1)`)**|`O(1)`|Python 內建|⚠ 空間消耗稍大|
|**`list` (`O(n)`)**|`O(n)`|小數據|❌ 查詢慢|

✅ **最佳解法：雙向鏈表 + 哈希表 (`O(1)`)，適用於所有場景！** 🚀

  

O