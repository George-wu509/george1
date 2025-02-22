Lintcode 143
给定一个有n个对象（包括k种不同的颜色，并按照1到k进行编号）的数组，将对象进行分类使相同颜色的对象相邻，并按照1,2，...k的顺序进行排序。

_**样例 1**_
```python
"""
输入: [3,2,2,1,4] 
4 
输出: [1,2,2,3,4]
```

```python
"""
输入: [2,1,1,2,2] 
2 
输出: [1,1,2,2,2]
```

```python
class Solution:

    def sort_colors2(self, colors: List[int], k: int):
        self.sort(colors, 1, k, 0, len(colors) - 1)
        
    def sort(self, colors, color_from, color_to, index_from, index_to):
        if color_from == color_to or index_from == index_to:
            return
            
        color = (color_from + color_to) // 2
        
        left, right = index_from, index_to
        while left <= right:
            while left <= right and colors[left] <= color:
                left += 1
            while left <= right and colors[right] > color:
                right -= 1
            if left <= right:
                colors[left], colors[right] = colors[right], colors[left]
                left += 1
                right -= 1
        
        self.sort(colors, color_from, color, index_from, right)
        self.sort(colors, color + 1, color_to, left, index_to)
```
pass

# **LintCode 143: Sort Colors II 解法分析**

## **解題目標**

給定一個陣列 `colors`，其中包含 `k` 種不同的顏色（數字 `1` 到 `k`），要求 **就地排序 (in-place sort)**，使陣列內的顏色按照數字大小排列。

這題是 **經典的多重 QuickSort (Rainbow Sort) 問題**，因為有 `k` 種顏色，而不是單純的 `0, 1, 2`（Sort Colors I）。

---

## **解法核心**

這裡使用的是 **基於 QuickSort 的「顏色排序」算法 (Rainbow Sort)**，思路類似於 **快速排序 (QuickSort)**：

1. **選擇中位顏色 `color = (color_from + color_to) // 2`** 作為基準。
2. **雙指針 Partition (劃分區間)**：
    - **左指針 (`left`)：** 尋找 **大於** `color` 的元素。
    - **右指針 (`right`)：** 尋找 **小於等於** `color` 的元素。
    - 若找到錯放的元素，則交換，使得 **左半部 ≤ `color`，右半部 > `color`**。
3. **遞迴對兩個部分排序**：
    - **左半部：** `color_from ~ color`
    - **右半部：** `color + 1 ~ color_to`
    - 直到每一種顏色都獨立排序。

---

## **為何這樣解？**

1. **避免使用 `O(k * n)` 計數排序**：
    - 若 `k` 很大，**計數排序 (Counting Sort)** 會耗費大量時間和空間 (`O(k + n)`)。
2. **避免 `O(n log n)` 標準快速排序**：
    - 若 `k` 很小（如 `k = 3`），標準 QuickSort 沒有利用數字範圍 `1 ~ k` 來優化。
3. **使用「顏色中位數 QuickSort」降低遞迴深度**：
    - **原本 QuickSort 遞迴深度為 `O(log n)`，但這裡遞迴次數僅 `O(log k)`**，因為每次只針對 `k` 個顏色分類。

---

## **解法步驟**

1. **初始化：** `sort(colors, 1, k, 0, len(colors) - 1)`
    - `color_from = 1, color_to = k`
    - `index_from = 0, index_to = n - 1`
2. **Partition (雙指針劃分區間)**
    - 以 `color = (color_from + color_to) // 2` 為基準。
    - **雙指針 `left` & `right` 遍歷陣列：**
        - `left` 找大於 `color` 的數
        - `right` 找小於等於 `color` 的數
        - 若找到錯誤的數則交換
    - 劃分結果：`左半部 ≤ color`，`右半部 > color`
3. **遞迴對兩個部分排序**
    - **左半部遞迴處理：** `sort(colors, color_from, color, index_from, right)`
    - **右半部遞迴處理：** `sort(colors, color + 1, color_to, left, index_to)`
4. **終止條件**
    - `color_from == color_to` (只剩一種顏色，不需要排序)
    - `index_from == index_to` (只剩一個數，不需要排序)

---

## **變數定義**

|變數名稱|作用|
|---|---|
|`colors`|存儲顏色數字的陣列|
|`color_from`|當前排序的最小顏色|
|`color_to`|當前排序的最大顏色|
|`index_from`|當前排序的最左索引|
|`index_to`|當前排序的最右索引|
|`color`|`color_from` 和 `color_to` 的中位數|
|`left`|指向錯放的大於 `color` 的數|
|`right`|指向錯放的小於等於 `color` 的數|

---

## **具體範例**

### **範例 1**

text

複製編輯

`輸入: colors = [3, 2, 2, 1, 4], k = 4`

#### **Step 1: Partition (第一層)**

text

複製編輯

`選擇 color = (1 + 4) // 2 = 2`

|原始數組|**左右指針移動**|結果|
|---|---|---|
|`[3, 2, 2, 1, 4]`|`left=0, right=4`|交換 `3` 和 `1`|
|`[1, 2, 2, 3, 4]`|`left=3, right=2`|分區完成|

- **左半部:** `[1, 2, 2]`
- **右半部:** `[3, 4]`

#### **Step 2: 遞迴對左半部 `[1, 2, 2]` 排序**

text

複製編輯

`選擇 color = (1 + 2) // 2 = 1`

|原始數組|**左右指針移動**|結果|
|---|---|---|
|`[1, 2, 2]`|`left=0, right=0`|分區完成|

- **左半部:** `[1]`
- **右半部:** `[2, 2]`（已排序）

#### **Step 3: 遞迴對右半部 `[3, 4]` 排序**

text

複製編輯

`選擇 color = (3 + 4) // 2 = 3`

|原始數組|**左右指針移動**|結果|
|---|---|---|
|`[3, 4]`|`left=1, right=0`|分區完成|

- **左半部:** `[3]`
- **右半部:** `[4]`（已排序）

最終結果：

text

複製編輯

`[1, 2, 2, 3, 4]`

---

## **時間與空間複雜度分析**

### **時間複雜度**

|操作|複雜度|說明|
|---|---|---|
|Partition|`O(n)`|每一層遞迴遍歷一次陣列|
|遞迴層數|`O(log k)`|每次將 `k` 種顏色劃分為兩半|
|**總計**|`O(n log k)`|快速排序風格劃分，`O(n log k)`|

### **空間複雜度**

- **遞迴棧深度：** `O(log k)`。
- **原地排序，沒有額外陣列，空間複雜度 `O(1)`**。

---

## **其他解法 (不寫 Code)**

1. **計數排序 (Counting Sort, `O(n + k)`)**
    
    - **適用情境**：當 `k` **較小**（如 `k ≈ 10⁴`）。
    - **方法**：
        - 用 `count[k]` 記錄每種顏色出現次數。
        - 遍歷 `count`，按順序填充 `colors` 陣列。
    - **缺點**：若 `k` 很大，會 **浪費大量空間**。
2. **桶排序 (Bucket Sort, `O(n)`)**
    
    - **適用情境**：當 `k` **較大**。
    - **方法**：
        - 用 `k` 個桶來存儲對應的顏色數字。
        - 遍歷 `colors`，將數字放入對應的桶。
        - 重新輸出排序後的數字。

---

## **總結**

|**解法**|**時間複雜度**|**適用場景**|**優缺點**|
|---|---|---|---|
|**雙指針 QuickSort (`O(n log k)`)**|`O(n log k)`|**適用於所有 `k` 值**|✅ 原地排序，適合大 `k`|
|**計數排序 (`O(n + k)`)**|`O(n + k)`|適合 `k` 小的情況|❌ `k` 大時空間浪費|
|**桶排序 (`O(n)`)**|`O(n)`|`k` 很大時|❌ 額外空間需求|

✅ **最佳解法：QuickSort (`O(n log k)`)，適用於所有 `k` 值，且空間複雜度 `O(1)`** 🚀

