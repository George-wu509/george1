Lintcode 122
给出的`n`个非负整数表示每个直方图的高度，每个直方图的宽均为1，在直方图中找到最大的矩形面积。

**样例 1：**
输入：
```python
"""
height = [2,1,5,6,2,3]
```
输出：
```python
"""
10
```
解释：
第三个和第四个直方图截取矩形面积为2*5=10。

**样例 2：**
输入：
```
height = [1,1]
```
输出：
```
2
```
解释：
第一个和第二个直方图截取矩形面积为2*1=2。


```python
    def largest_rectangle_area(self, heights):
        indices_stack = []
        area = 0
        for index, height in enumerate(heights + [0]):
            while indices_stack and heights[indices_stack[-1]] >= height:		#如果列表尾部高度大于当前高度
                popped_index = indices_stack.pop()
                left_index = indices_stack[-1] if indices_stack else -1		
                width = index - left_index - 1		#如果列表为空，则宽度为index，否则为index-indices_stack[-1]-1
                area = max(area, width * heights[popped_index])
                
            indices_stack.append(index)		#压入列表中
            
        return area
```
pass
题解：  
用九章算法强化班中讲过的单调栈(stack)。维护一个单调递增栈，逐个将元素 push 到栈里。push 进去之前先把 >= 自己的元素 pop 出来。  
每次从栈中 pop 出一个数的时候，就找到了往左数比它小的第一个数（当前栈顶）和往右数比它小的第一个数（即将入栈的数），  
从而可以计算出这两个数中间的部分宽度 * 被pop出的数，就是以这个被pop出来的数为最低的那个直方向两边展开的最大矩阵面积。  
因为要计算两个数中间的宽度，因此放在 stack 里的是每个数的下标。

### **LintCode 122 - Largest Rectangle in Histogram**

#### **解法分析**

本題的目標是找到 **柱狀圖中能形成的最大矩形面積**。給定一個非負整數陣列 `heights`，每個元素代表對應柱狀圖的高度，求可形成的最大矩形面積。

---

### **解法思路**

1. **使用單調遞增棧 (Monotonic Increasing Stack)** 來維護柱狀圖的高度索引：
    
    - 棧內存儲 **柱子的索引值**，確保棧內元素對應的高度是 **單調遞增** 的。
    - 一旦遇到當前高度小於棧頂對應的高度，則表示棧頂元素的最大擴展範圍已經確定，可以計算面積。
2. **處理方式**
    
    - 為了統一處理，我們 **在 `heights` 陣列後面額外添加 `0`**，確保最後一個柱子能夠被正確計算。
    - 遍歷 `heights`，當發現 **當前高度小於棧頂對應的高度時**：
        - 彈出棧頂索引，並計算對應的矩形面積。
        - 矩形的 `width` 由當前索引 `index` 和棧內的前一個索引決定：
            - `width = index - left_index - 1`，其中 `left_index` 是彈出索引的左邊界（如果棧為空，則 `left_index = -1`）。
        - 持續更新 `area`。
    - 最後，將當前索引 `index` 入棧。

---

### **變數說明**

|變數名稱|說明|
|---|---|
|`heights`|柱狀圖高度數組|
|`indices_stack`|單調遞增棧，存儲柱狀圖的索引|
|`area`|最大矩形面積的結果|
|`index`|當前遍歷的柱狀圖索引|
|`height`|`heights[index]` 當前柱子的高度|
|`popped_index`|被彈出的柱狀圖索引（該索引的矩形高度已確定）|
|`left_index`|`popped_index` 的左邊界索引（用來計算矩形寬度）|
|`width`|計算當前柱狀圖能形成的最大矩形的寬度|

---

### **範例**

#### **輸入**

python

複製編輯

`heights = [2, 1, 5, 6, 2, 3]`

#### **處理流程**

1. **初始狀態**

    `heights = [2, 1, 5, 6, 2, 3, 0]  # 末尾加上 0 確保計算 indices_stack = [] area = 0`
    
2. **遍歷 `heights`**
    
    - `index = 0, height = 2`，`indices_stack = [0]`
    - `index = 1, height = 1`
        - `heights[indices_stack[-1]] = 2 > 1`，彈出 `0`
        - `width = 1 - (-1) - 1 = 1`
        - `area = max(0, 2 * 1) = 2`
        - `indices_stack = [1]`
    - `index = 2, height = 5`，`indices_stack = [1, 2]`
    - `index = 3, height = 6`，`indices_stack = [1, 2, 3]`
    - `index = 4, height = 2`
        - `heights[indices_stack[-1]] = 6 > 2`，彈出 `3`
        - `width = 4 - 2 - 1 = 1`
        - `area = max(2, 6 * 1) = 6`
        - `heights[indices_stack[-1]] = 5 > 2`，彈出 `2`
        - `width = 4 - 1 - 1 = 2`
        - `area = max(6, 5 * 2) = 10`
        - `indices_stack = [1, 4]`
    - `index = 5, height = 3`，`indices_stack = [1, 4, 5]`
    - `index = 6, height = 0`
        - `heights[indices_stack[-1]] = 3 > 0`，彈出 `5`
        - `width = 6 - 4 - 1 = 1`
        - `area = max(10, 3 * 1) = 10`
        - `heights[indices_stack[-1]] = 2 > 0`，彈出 `4`
        - `width = 6 - 1 - 1 = 4`
        - `area = max(10, 2 * 4) = 10`
        - `heights[indices_stack[-1]] = 1 > 0`，彈出 `1`
        - `width = 6 - (-1) - 1 = 6`
        - `area = max(10, 1 * 6) = 10`
        - `indices_stack = [6]`
3. **最終結果**
    
    複製編輯
    
    `最大矩形面積 = 10`
    

#### **輸出**

python

複製編輯

`10`

---

### **時間與空間複雜度分析**

#### **時間複雜度**

- `heights` 遍歷一次，每個元素最多 **進棧一次、出棧一次**，所以時間複雜度為 **O(N)**。

#### **空間複雜度**

- 使用 `indices_stack` 儲存最多 `N` 個索引，最壞情況下棧長度為 **O(N)**。
- 其餘變數 `area`、`index`、`height` 皆為 **O(1)**。
- **總體空間複雜度為 O(N)**。

---

### **其他解法想法**

4. **暴力解法 (O(N²))**
    
    - 針對每個 `heights[i]`，往左右展開，找到 `heights[i]` 能延伸的最大範圍，計算面積並更新最大值。
    - **時間複雜度 O(N²)**，適用於 `N` 很小的情況。
5. **分治法 (O(N log N))**
    
    - 遞迴尋找 `heights` 中的最小元素，將問題拆分為左右兩部分，對每部分重複計算最大矩形。
    - **時間複雜度 O(N log N)**，適用於特定數組。
6. **線段樹 (O(N log N))**
    
    - 構建線段樹，每次查詢某區間內的最小值，然後透過遞迴方法尋找最大矩形面積。
    - **時間複雜度 O(N log N)**。

---

### **總結**

- **最優解法：單調棧 (O(N) 時間, O(N) 空間)**。
- **暴力解法適用於小數據 (O(N²))**。
- **若 `heights` 結構性較好，可用分治法或線段樹 (O(N log N))**。