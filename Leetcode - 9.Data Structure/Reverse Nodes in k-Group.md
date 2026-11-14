
LeetCode 25 / LintCode 450 (NeetCode 150)
給你一個鏈結串列的頭節點 `head` ，每 `k` 個節點一組進行翻轉，請你返回修改後的鏈結串列。
`k` 是一個正整數，它的值小於或等於鏈結串列的長度。如果節點總數不是 `k` 的整數倍，那麼請將最後剩餘的節點保持原有順序。
你不能只是單純的改變節點內部的值，而是需要實際進行節點交換。

**樣例 1：**
輸入：
```python
head = [1, 2, 3, 4, 5], k = 2
```
輸出：
```python
[2, 1, 4, 3, 5]
```
解釋：
第一組 `[1, 2]` 翻轉為 `[2, 1]`
第二組 `[3, 4]` 翻轉為 `[4, 3]`
剩餘 `[5]` 不足 2 個，保持原樣

**樣例 2：**
輸入：
```python
head = [1, 2, 3, 4, 5], k = 3
```
輸出：
```python
[3, 2, 1, 4, 5]
```
解釋：
第一組 `[1, 2, 3]` 翻轉為 `[3, 2, 1]`
剩餘 `[4, 5]` 不足 3 個，保持原樣

**樣例 3：**
輸入：
```python
head = [1, 2, 3, 4], k = 1
```
輸出：
```python
[1, 2, 3, 4]
```
解釋：
每 1 個一組翻轉，串列維持不變

---

### 解法一：遞迴法（直觀好寫，空間 O(n/k)）

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        curr = head
        count = 0
        # 1. 先檢查剩餘節點是否有滿 k 個
        while curr and count < k:
            curr = curr.next
            count += 1
        
        # 2. 若有滿 k 個，則進行翻轉並遞迴處理後續串列
        if count == k:
            # 遞迴翻轉剩下的部分，回傳值為下一組翻轉後的新頭節點
            reversed_head = self.reverseKGroup(curr, k)
            
            # 將當前這 k 個節點逐一反轉，並接上 reversed_head
            while count > 0:
                nxt = head.next
                head.next = reversed_head
                reversed_head = head
                head = nxt
                count -= 1
            head = reversed_head
            
        return head
```
pass

### 解法二：迭代法 + Dummy Node（面試最佳解，空間 O(1)）

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        groupPrev = dummy

        while True:
            # 1. 找到當前這組的第 k 個節點
            kth = self.getKth(groupPrev, k)
            if not kth:
                break
            groupNext = kth.next

            # 2. 翻轉當前這一組 (從 groupPrev.next 到 kth)
            prev, curr = kth.next, groupPrev.next
            while curr != groupNext:
                tmp = curr.next
                curr.next = prev
                prev = curr
                curr = tmp

            # 3. 重新接回主鏈結串列，並移動 groupPrev 到下一組的前驅位置
            tmp = groupPrev.next   # 原本這組的第一個節點，翻轉後變成最後一個
            groupPrev.next = kth   # groupPrev 指向翻轉後的新頭節點 (kth)
            groupPrev = tmp        # 更新 groupPrev 為下一組的前一個節點

        return dummy.next

    def getKth(self, curr: ListNode, k: int) -> Optional[ListNode]:
        while curr and k > 0:
            curr = curr.next
            k -= 1
        return curr
```

解釋:
step1: 建立 `dummy` 節點指向 `head`，並令 `groupPrev = dummy`（代表每一組「前一個」節點）。
step2: 假設 `head = [1, 2, 3, 4, 5], k = 2`。
step3: 第一輪迴圈：`getKth(dummy, 2)` 找到 `kth = 2`，記錄下一組開頭 `groupNext = 3`。
step4: 反轉區間 `[1, 2]`，技巧是將 `prev` 初始設為 `groupNext (3)`，這樣反轉完 `1` 會直接指向 `3`（變成 `2 -> 1 -> 3`）。
step5: 調整外部指標：`dummy.next` 改指向 `kth (2)`，並將 `groupPrev` 更新為原本組內的頭 `1`（現在是這組的尾巴）。此時串列為 `dummy -> 2 -> 1 -> 3 -> 4 -> 5`。
step6: 第二輪迴圈：`getKth(1, 2)` 找到 `kth = 4`，`groupNext = 5`，反轉 `[3, 4]` 並接上，串列變為 `dummy -> 2 -> 1 -> 4 -> 3 -> 5`，`groupPrev` 移至 `3`。
step7: 第三輪迴圈：`getKth(3, 2)` 走到 `None`（不足 2 個），觸發 `break`，回傳 `dummy.next` 即為 `2`。

---

### **LeetCode 25 - Reverse Nodes in k-Group（K 個一組翻轉鏈結串列）**

這道題目是鏈結串列操作中的經典 **Hard** 題型，本質上是 **「反轉鏈結串列（LeetCode 206）」** 與 **「反轉指定區間鏈結串列（LeetCode 92）」** 的進階綜合題，非常考驗指標（Pointer）操作與邊界條件的處理能力。

---

## **解法解析**

這道題的核心在於**「分組定位」**與**「局部反轉後的頭尾重連」**。

### **思路**

1. **引入 Dummy Node（虛擬頭節點）**
   - 因為原本的頭節點 `head` 在第一組翻轉後一定會改變位置（變成第 `k` 個節點），使用 `dummy -> head` 可以統一所有分組的操作邏輯，避免額外處理頭節點的特殊情況。
2. **定位邊界指標（四個關鍵位置）**
   - `groupPrev`：當前要翻轉這一組的**前一個節點**。
   - `groupStart`（即 `groupPrev.next`）：當前這一組翻轉前的**第一個節點**（翻轉後會變成這組的**尾巴**）。
   - `kth`：當前這一組的**第 `k` 個節點**（翻轉後會變成這組的**新頭節點**）。
   - `groupNext`（即 `kth.next`）：下一組的**第一個節點**。
3. **局部反轉的巧妙接法**
   - 在反轉 `[groupStart ... kth]` 時，直接把 `prev` 初始化為 `groupNext`，而不是 `None`。
   - 這樣一來，當第一個節點 `groupStart` 執行 `curr.next = prev` 時，它就會自動連向下一組的開頭 `groupNext`，省去事後再找尾巴接線的步驟。
4. **重新連接與推進**
   - 翻轉完成後，`kth` 成為新組頭，因此 `groupPrev.next = kth`。
   - 原本的 `groupStart`（先用 `tmp` 存起來）現在位於這組的最後一個，正好成為下一組的 `groupPrev`。

---

## **具體舉例**

假設 `head = [1, 2, 3, 4, 5], k = 2`，迭代過程的指標與串列狀態變化如下：

| 輪次 | `groupPrev` | `kth` | `groupNext` | 局部翻轉後狀態 | 完整串列狀態 (`dummy -> ...`) |
|---|---|---|---|---|---|
| 初始化 | `dummy (0)` | `-` | `-` | `-` | `0 -> 1 -> 2 -> 3 -> 4 -> 5` |
| 第 1 組 | `dummy (0)` | `2` | `3` | `2 -> 1 -> 3` | `0 -> 2 -> 1 -> 3 -> 4 -> 5` |
| 第 2 組 | `1` | `4` | `5` | `4 -> 3 -> 5` | `0 -> 2 -> 1 -> 4 -> 3 -> 5` |
| 第 3 組 | `3` | `None` (不足 k) | `-` | 跳出迴圈 (`break`) | `0 -> 2 -> 1 -> 4 -> 3 -> 5` |

最終回傳 `dummy.next`，即 `[2, 1, 4, 3, 5]`。

---

## **時間與空間複雜度分析**

- **時間複雜度：O(n)**
  - `getKth` 尋找每組結尾總共遍歷所有節點一次（O(n)）。
  - 內部 `while` 迴圈反轉每個節點也總共只走訪每個節點一次（O(n)）。
  - 總操作次數約為 `2n`，因此時間複雜度為 **O(n)**。
- **空間複雜度：O(1)**
  - 迭代法只使用了 `dummy`、`groupPrev`、`kth`、`groupNext`、`prev`、`curr`、`tmp` 等常數個指標變數，沒有額外配置資料結構，因此空間複雜度為 **O(1)**。

---

## **其他解法**

### **1. 遞迴解法（Recursion）**
- **思路**：
  - 先用指標往後走 `k` 步確認數量足夠，接著對第 `k+1` 個節點呼叫 `reverseKGroup`，拿到後方已處理好的新串列頭節點，再把當前這 `k` 個節點頭插法接上去。
- **時間複雜度**：O(n)
- **空間複雜度**：O(n / k)（因為遞迴呼叫堆疊的深度為總組數 `n / k`）

### **2. 堆疊輔助法（Stack）**
- **思路**：
  - 每次將 `k` 個節點依序推進 `stack` 中；若湊滿 `k` 個，就依序 `pop()` 出來重新串接；若還沒湊滿 `k` 個串列就結束了，則直接保留原本的順序接在最後。
- **時間複雜度**：O(n)
- **空間複雜度**：O(k)（堆疊最多同時存放 `k` 個節點）

---

## **總結**

- **最直觀好寫的解法**：**遞迴法** 或 **Stack 輔助法**，邏輯清晰不容易寫錯指標，適合快速驗證思路。
- **面試最佳解（Follow-up O(1) 空間要求）**：**迭代法 + Dummy Node**，時間 O(n)，空間 O(1)。寫這題時只要掌握「每次先抓出 `groupPrev` 與 `groupNext`，把 `prev` 設為 `groupNext` 再反轉」的模板，就能大幅降低指標斷鏈的風險。

---

## **鏈結串列反轉類問題整理**

| 題號 | 題目名稱 | 問題類型 | 解法 |
|---|---|---|---|
| **LeetCode 206** | Reverse Linked List（反轉鏈結串列） | 整條單向鏈結串列完全反轉 | 雙指標迭代 `prev, curr`（時間 O(n)，空間 O(1)） |
| **LeetCode 92** | Reverse Linked List II（反轉鏈結串列 II） | 反轉指定區間 `[left, right]` 內的節點 | Dummy Node + 定位 `leftPrev` 後做 `right - left` 次頭插法 |
| **LeetCode 24** | Swap Nodes in Pairs（兩兩交換鏈結串列中的節點） | 每 2 個相鄰節點一組交換（即 `k = 2` 的特例） | Dummy Node + 三指標局部交換，或直接套用 `k = 2` 模板 |
| **LeetCode 25** | Reverse Nodes in k-Group（K 個一組翻轉鏈結串列） | 每 `k` 個節點一組反轉，不足 `k` 個保持原樣 | Dummy Node + 分組確認長度 + 局部反轉重連（空間 O(1)） |
| **LeetCode 61** | Rotate List（旋轉鏈結串列） | 將鏈結串列向右旋轉 `k` 個位置 | 先計算長度並連成環，再於 `len - (k % len)` 處斷開 |
| **LeetCode 143** | Reorder List（重排鏈結串列） | 將 `L0→L1→…→Ln` 重排為 `L0→Ln→L1→Ln-1…` | 快慢指標找中點 + 反轉後半段串列 + 交替合併兩條串列 |
| **LeetCode 234** | Palindrome Linked List（迴文鏈結串列） | 判斷單向鏈結串列是否為迴文結構 | 快慢指標找中點 + 反轉後半段串列 + 雙指標比對節點值 |

---

### **進階變形題：如果最後不足 `k` 個的剩餘節點「也要」一起翻轉呢？**

只需將原本找 `kth` 節點的 `getKth` 邏輯微調：只要還沒走到串列底端，就盡量往後走最多 `k` 步（不因為不足 `k` 步而回傳 `None` 跳出）：

```python
class Solution:
    def reverseKGroupAll(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        groupPrev = dummy

        while groupPrev.next:
            # 盡量往後找最多 k 個節點，若不足 k 個則停在最後一個非空節點
            kth = groupPrev
            steps = k
            while kth.next and steps > 0:
                kth = kth.next
                steps -= 1

            groupNext = kth.next

            # 翻轉從 groupPrev.next 到 kth 的所有節點（即便不足 k 個也會翻轉）
            prev, curr = kth.next, groupPrev.next
            while curr != groupNext:
                tmp = curr.next
                curr.next = prev
                prev = curr
                curr = tmp

            tmp = groupPrev.next
            groupPrev.next = kth
            groupPrev = tmp

        return dummy.next
```
````