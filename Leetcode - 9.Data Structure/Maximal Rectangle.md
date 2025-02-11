
LintCode 510
给你一个二维矩阵，权值为`False`和`True`，找到一个最大的矩形，使得里面的值全部为`True`，输出它的面积



---

### **題目分析**

在一個布爾矩陣（只包含 `0` 和 `1`）中，找到值為 `1` 的最大矩形面積。

---

### **解法核心思路**

該問題可以轉化為多次解決「柱狀圖中的最大矩形面積」問題：

1. 將每一行作為柱狀圖的基底，計算每一列的高度。
2. 對於每一行計算其對應的柱狀圖中最大矩形面積，並更新全局最大值。

這種解法的核心是 **單調遞增棧**，用於快速計算柱狀圖中最大的矩形面積。

---

### **解法步驟**

#### **1. 初始化**

- 檢查矩陣是否為空，若是則返回 `0`。
- 初始化 `max_rectangle` 為 `0`，用於存儲全局的最大矩形面積。
- 初始化一個 `heights` 列表，用於存儲柱狀圖中每列的高度。

---

#### **2. 遍歷矩陣每一行**

- 對於每一行的每個元素，更新對應列的高度：
    - 如果當前元素為 `1`，則將該列的高度加 `1`。
    - 如果當前元素為 `0`，則將該列的高度重置為 `0`。
- 計算該行的柱狀圖的最大矩形面積，並更新 `max_rectangle`。

---

#### **3. 計算柱狀圖最大矩形面積（`find_max_rectangle` 函數）**

- **單調棧法：**
    - 初始化 `indices_stack` 為空，用於存儲柱狀圖中高度的索引。
    - 遍歷 `heights` 列表，對每個高度進行如下操作：
        1. 當當前高度小於棧頂對應的高度時，從棧中彈出索引，計算以該高度為最小高度的矩形面積。
        2. 更新最大矩形面積。
        3. 將當前索引壓入棧。
    - 在遍歷結束後，附加一個高度為 `-1` 的柱子，以確保所有柱子都能從棧中彈出並處理。

---
Example:
```python
输入:
[
  [1, 1, 0, 0, 1],
  [0, 1, 0, 0, 1],
  [0, 0, 1, 1, 1],
  [0, 0, 1, 1, 1],
  [0, 0, 0, 0, 1]
]
输出: 6
```
```python
输入:
[
    [0,0],
    [0,0]
]
输出: 0
```

```python
class Solution:
    """
    @param matrix: a boolean 2D matrix
    @return: an integer
    """
    def maximalRectangle(self, matrix):
        if not matrix:
            return 0
            
        max_rectangle = 0
        heights = [0] * len(matrix[0])
        for row in matrix:
            for index, num in enumerate(row):
                heights[index] = heights[index] + 1 if num else 0
            max_rectangle = max(
                max_rectangle,
                self.find_max_rectangle(heights),
            )

        return max_rectangle

    def find_max_rectangle(self, heights):
        indices_stack = []
        max_rectangle = 0
        for index, height in enumerate(heights + [-1]):
            while indices_stack and heights[indices_stack[-1]] >= height:
                popped = indices_stack.pop(-1)
                left_bound = indices_stack[-1] if indices_stack else -1
                max_rectangle = max(
                    max_rectangle,
                    (index - left_bound - 1) * heights[popped],
                )
            indices_stack.append(index)
            print(indices_stack)
        
        return max_rectangle
```
pass

### **具體舉例**

#### 輸入：

`matrix = [     [1, 0, 1, 0, 0],     [1, 0, 1, 1, 1],     [1, 1, 1, 1, 1],     [1, 0, 0, 1, 0] ]`

#### 步驟詳解：

1. **初始化：**
    
    - `max_rectangle = 0`
    - `heights = [0, 0, 0, 0, 0]`
2. **處理第一行 `[1, 0, 1, 0, 0]`：**
    
    - 更新 `heights = [1, 0, 1, 0, 0]`
    - 計算柱狀圖最大矩形面積：
        - 單調棧操作：
            - 壓入索引 `0`（高度 `1`）。
            - 壓入索引 `2`（高度 `1`）。
            - 結算棧中所有柱子，最大面積為 `1`。
    - `max_rectangle = 1`
3. **處理第二行 `[1, 0, 1, 1, 1]`：**
    
    - 更新 `heights = [2, 0, 2, 1, 1]`
    - 計算柱狀圖最大矩形面積：
        - 單調棧操作：
            - 壓入索引 `0`（高度 `2`）。
            - 壓入索引 `2`（高度 `2`）。
            - 壓入索引 `3`（高度 `1`）。
            - 壓入索引 `4`（高度 `1`）。
            - 結算棧中所有柱子，最大面積為 `3`。
    - `max_rectangle = 3`
4. **處理第三行 `[1, 1, 1, 1, 1]`：**
    
    - 更新 `heights = [3, 1, 3, 2, 2]`
    - 計算柱狀圖最大矩形面積：
        - 單調棧操作：
            - 壓入索引 `0`（高度 `3`）。
            - 壓入索引 `1`（高度 `1`）。
            - 壓入索引 `2`（高度 `3`）。
            - 壓入索引 `3`（高度 `2`）。
            - 壓入索引 `4`（高度 `2`）。
            - 結算棧中所有柱子，最大面積為 `6`。
    - `max_rectangle = 6`
5. **處理第四行 `[1, 0, 0, 1, 0]`：**
    
    - 更新 `heights = [4, 0, 0, 3, 0]`
    - 計算柱狀圖最大矩形面積：
        - 單調棧操作：
            - 壓入索引 `0`（高度 `4`）。
            - 壓入索引 `3`（高度 `3`）。
            - 結算棧中所有柱子，最大面積仍為 `6`。
    - `max_rectangle = 6`

---

### **最終結果**

最大矩形面積為 `6`。

---

### **時間與空間複雜度分析**

1. **時間複雜度：**
    
    - 遍歷矩陣的每一行，共 O(m)O(m)O(m) 行，其中 mmm 為矩陣行數。
    - 每行調用一次單調棧法處理柱狀圖，時間複雜度為 O(n)O(n)O(n)，其中 nnn 為矩陣列數。
    - 總時間複雜度為 O(m⋅n)O(m \cdot n)O(m⋅n)。
2. **空間複雜度：**
    
    - 使用一個大小為 nnn 的 `heights` 列表和一個棧，空間複雜度為 O(n)O(n)O(n)。

---

### **其他解法簡述**

1. **暴力解法：**
    
    - 對矩陣的每一個矩形子區域進行遍歷並檢查是否全部為 `1`，計算其面積。
    - 時間複雜度：O((m⋅n)2)O((m \cdot n)^2)O((m⋅n)2)。
2. **動態規劃：**
    
    - 使用動態規劃計算以每個位置為右下角的矩形面積，並記錄最大值。
    - 時間複雜度：O(m⋅n)O(m \cdot n)O(m⋅n)。